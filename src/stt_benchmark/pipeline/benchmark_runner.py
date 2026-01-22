"""Benchmark runner for STT services using Pipecat pipeline."""

import asyncio
from collections.abc import Callable
from pathlib import Path

import aiohttp
from loguru import logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from stt_benchmark.config import get_config
from stt_benchmark.models import AudioSample, BenchmarkResult, ServiceName
from stt_benchmark.observers.metrics_collector import MetricsCollectorObserver
from stt_benchmark.observers.transcription_collector import TranscriptionCollectorObserver
from stt_benchmark.pipeline.service_factory import create_stt_service, service_needs_aiohttp
from stt_benchmark.pipeline.synthetic_transport import SyntheticInputTransport


class BenchmarkRunner:
    """Runs STT benchmarks using Pipecat pipeline with observers."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_ms: int = 20,
        vad_stop_secs: float = 0.2,
        max_silence_timeout_secs: float = 10.0,
        transcription_timeout_secs: float = 10.0,
    ):
        """Initialize the benchmark runner.

        Args:
            sample_rate: Audio sample rate in Hz.
            chunk_ms: Duration of each audio chunk in ms.
            vad_stop_secs: Silence duration for VAD stop.
            max_silence_timeout_secs: Max time to send silence while waiting for transcription.
            transcription_timeout_secs: Max time to wait for transcription after silence ends.
        """
        config = get_config()
        self.sample_rate = sample_rate or config.sample_rate
        self.chunk_ms = chunk_ms or config.chunk_duration_ms
        self.vad_stop_secs = vad_stop_secs or config.vad_stop_secs
        self.max_silence_timeout_secs = max_silence_timeout_secs or config.max_silence_timeout_secs
        self.transcription_timeout_secs = (
            transcription_timeout_secs or config.transcription_timeout_secs
        )

    async def benchmark_sample(
        self,
        sample: AudioSample,
        service_name: ServiceName,
        model: str | None = None,
    ) -> BenchmarkResult:
        """Benchmark a single audio sample with an STT service.

        Args:
            sample: The audio sample to benchmark.
            service_name: The STT service to use.
            model: Optional model name override.

        Returns:
            BenchmarkResult with TTFB and transcription.
        """
        # Load audio data
        audio_path = Path(sample.audio_path)
        if not audio_path.exists():
            return BenchmarkResult(
                sample_id=sample.sample_id,
                service_name=service_name,
                model_name=model,
                audio_duration_seconds=sample.duration_seconds,
                error=f"Audio file not found: {audio_path}",
            )

        audio_data = audio_path.read_bytes()

        # Set up observers
        metrics_observer = MetricsCollectorObserver()
        transcription_observer = TranscriptionCollectorObserver()

        metrics_observer.set_current_sample(sample.sample_id)
        transcription_observer.set_current_sample(sample.sample_id)

        # Create aiohttp session if needed
        session: aiohttp.ClientSession | None = None
        if service_needs_aiohttp(service_name):
            session = aiohttp.ClientSession()

        try:
            # Create STT service
            stt_service = await create_stt_service(
                service_name,
                aiohttp_session=session,
                model=model,
            )

            # Create transport with audio
            # Pass transcription_received event so transport sends silence
            # until transcription arrives (or timeout)
            transport = SyntheticInputTransport(
                audio_data=audio_data,
                sample_rate=self.sample_rate,
                chunk_ms=self.chunk_ms,
                vad_stop_secs=self.vad_stop_secs,
                transcription_received=transcription_observer._transcription_received,
                max_silence_timeout=self.max_silence_timeout_secs,
            )

            # Build pipeline
            pipeline = Pipeline([transport, stt_service])

            # Create task with observers
            task = PipelineTask(
                pipeline,
                params=PipelineParams(
                    audio_in_sample_rate=self.sample_rate,
                    audio_in_channels=1,
                    enable_metrics=True,
                ),
                observers=[metrics_observer, transcription_observer],
            )

            # Run pipeline
            runner = PipelineRunner(handle_sigint=False)
            pipeline_coro = runner.run(task)
            pipeline_task = asyncio.create_task(pipeline_coro)

            try:
                # Wait for audio to complete
                await transport.wait_for_audio_complete(timeout=60.0)

                # Wait for first transcription with timeout
                try:
                    transcription = await transcription_observer.wait_for_transcription(
                        timeout=self.transcription_timeout_secs
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{service_name.value}] Transcription timeout after {self.transcription_timeout_secs}s"
                    )
                    # Still try to get partial transcription if any
                    transcription = transcription_observer.get_transcription_for_sample(
                        sample.sample_id
                    )
                    if not transcription:
                        raise  # Re-raise if no transcription at all
                else:
                    # Get the final concatenated transcription
                    transcription = transcription_observer.get_transcription_for_sample(
                        sample.sample_id
                    )

                # Wait for TTFB metric (services without finalized transcripts
                # use a 2-second timeout in Pipecat to determine final transcript)
                ttfb = metrics_observer.get_ttfb_for_sample(sample.sample_id)
                if ttfb is None:
                    logger.debug(
                        f"[{service_name.value}] Waiting for TTFB metric (timeout-based services)..."
                    )
                    await metrics_observer.wait_for_ttfb(timeout=2.5)

            finally:
                # Cancel the pipeline task
                await task.cancel()
                try:
                    await pipeline_task
                except asyncio.CancelledError:
                    pass

            # Get TTFB from observer
            ttfb = metrics_observer.get_ttfb_for_sample(sample.sample_id)

            logger.debug(
                f"[{service_name.value}] Sample {sample.sample_id}: " f"TTFB={ttfb:.3f}s"
                if ttfb
                else "TTFB=N/A"
            )

            return BenchmarkResult(
                sample_id=sample.sample_id,
                service_name=service_name,
                model_name=model,
                ttfb_seconds=ttfb,
                transcription=transcription,
                audio_duration_seconds=sample.duration_seconds,
            )

        except Exception as e:
            logger.error(f"[{service_name.value}] Error benchmarking {sample.sample_id}: {e}")
            return BenchmarkResult(
                sample_id=sample.sample_id,
                service_name=service_name,
                model_name=model,
                audio_duration_seconds=sample.duration_seconds,
                error=str(e),
            )

        finally:
            # Cleanup aiohttp session
            if session:
                await session.close()

    async def benchmark_batch(
        self,
        samples: list[AudioSample],
        service_name: ServiceName,
        model: str | None = None,
        progress_callback: Callable | None = None,
    ) -> list[BenchmarkResult]:
        """Benchmark multiple audio samples sequentially.

        Args:
            samples: List of audio samples to benchmark.
            service_name: The STT service to use.
            model: Optional model name override.
            progress_callback: Optional callback(current, total, sample_id).

        Returns:
            List of BenchmarkResult objects.
        """
        results = []

        for i, sample in enumerate(samples):
            if progress_callback:
                progress_callback(i, len(samples), sample.sample_id)

            result = await self.benchmark_sample(sample, service_name, model)
            results.append(result)

            # Brief delay between samples to avoid rate limiting
            await asyncio.sleep(0.1)

        if progress_callback:
            progress_callback(len(samples), len(samples), "complete")

        return results
