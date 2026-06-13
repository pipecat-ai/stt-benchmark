"""Benchmark runner for STT services using Pipecat pipeline."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.observers.base_observer import BaseObserver

if TYPE_CHECKING:
    import aiohttp

from stt_benchmark.config import get_config
from stt_benchmark.models import AudioSample, BenchmarkResult, ServiceName
from stt_benchmark.observers.metrics_collector import MetricsCollectorObserver
from stt_benchmark.observers.transcription_collector import TranscriptionCollectorObserver
from stt_benchmark.pipeline.synthetic_transport import SyntheticInputTransport
from stt_benchmark.services import GrpcServiceOptions, create_stt_service, get_service_definition


class BenchmarkRunner:
    """Runs STT benchmarks using Pipecat pipeline with observers."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_ms: int = 20,
        vad_stop_secs: float = 0.2,
        max_silence_timeout_secs: float = 10.0,
        transcription_timeout_secs: float = 10.0,
        post_transcription_delay_secs: float = 2.0,
        grpc_options: GrpcServiceOptions | None = None,
    ):
        """Initialize the benchmark runner.

        Args:
            sample_rate: Audio sample rate in Hz.
            chunk_ms: Duration of each audio chunk in ms.
            vad_stop_secs: Silence duration for VAD stop.
            max_silence_timeout_secs: Max time to send silence while waiting for transcription.
            transcription_timeout_secs: Max time to wait for transcription after silence ends.
            post_transcription_delay_secs: Time to continue sending silence after first
                transcription to collect additional segments.
            grpc_options: Runtime options for gRPC-based ASR services.
        """
        config = get_config()
        self.sample_rate = sample_rate or config.sample_rate
        self.chunk_ms = chunk_ms or config.chunk_duration_ms
        self.vad_stop_secs = vad_stop_secs or config.vad_stop_secs
        self.max_silence_timeout_secs = max_silence_timeout_secs or config.max_silence_timeout_secs
        self.transcription_timeout_secs = (
            transcription_timeout_secs or config.transcription_timeout_secs
        )
        self.post_transcription_delay_secs = post_transcription_delay_secs
        self.grpc_options = grpc_options

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
        # Load audio data (resample to runner sample rate when needed)
        audio_path = Path(sample.audio_path)
        if not audio_path.exists():
            return BenchmarkResult(
                sample_id=sample.sample_id,
                service_name=service_name,
                model_name=model,
                audio_duration_seconds=sample.duration_seconds,
                error=f"Audio file not found: {audio_path}",
            )

        from stt_benchmark.debug.audio_source import load_audio_file

        audio_data, _ = load_audio_file(
            audio_path,
            sample_rate=self.sample_rate,
        )

        # Set up observers
        metrics_observer = MetricsCollectorObserver()
        transcription_observer = TranscriptionCollectorObserver()

        metrics_observer.set_current_sample(sample.sample_id)
        transcription_observer.set_current_sample(sample.sample_id)

        try:
            # Check if this service needs an aiohttp session
            definition = get_service_definition(service_name.value)

            if definition.needs_aiohttp:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    return await self._run_pipeline(
                        sample=sample,
                        service_name=service_name,
                        model=model,
                        audio_data=audio_data,
                        metrics_observer=metrics_observer,
                        transcription_observer=transcription_observer,
                        aiohttp_session=session,
                    )
            else:
                return await self._run_pipeline(
                    sample=sample,
                    service_name=service_name,
                    model=model,
                    audio_data=audio_data,
                    metrics_observer=metrics_observer,
                    transcription_observer=transcription_observer,
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

    async def debug_sample(
        self,
        sample: AudioSample,
        service_name: ServiceName,
        model: str | None = None,
        *,
        source_label: str | None = None,
        audio_data: bytes | None = None,
    ) -> BenchmarkResult:
        """Run one sample through the benchmark pipeline with stderr debug tracing."""
        from stt_benchmark.observers.debug_trace import DebugTraceObserver

        if audio_data is None:
            audio_path = Path(sample.audio_path)
            if not audio_path.exists():
                result = BenchmarkResult(
                    sample_id=sample.sample_id,
                    service_name=service_name,
                    model_name=model,
                    audio_duration_seconds=sample.duration_seconds,
                    error=f"Audio file not found: {audio_path}",
                )
                debug_observer = DebugTraceObserver()
                debug_observer.log_start(
                    service=service_name.value,
                    chunk_ms=self.chunk_ms,
                    vad_stop_secs=self.vad_stop_secs,
                    audio_bytes=0,
                    duration_seconds=sample.duration_seconds,
                    source_label=source_label or sample.sample_id,
                )
                debug_observer.log_done(
                    transcription=None,
                    ttfb_seconds=None,
                    error=result.error,
                )
                return result
            from stt_benchmark.debug.audio_source import load_audio_file

            audio_data, duration_seconds = load_audio_file(
                audio_path,
                sample_rate=self.sample_rate,
            )
            sample = AudioSample(
                sample_id=sample.sample_id,
                audio_path=sample.audio_path,
                duration_seconds=duration_seconds,
                language=sample.language,
                dataset_index=sample.dataset_index,
            )
        debug_observer = DebugTraceObserver()
        debug_observer.log_start(
            service=service_name.value,
            chunk_ms=self.chunk_ms,
            vad_stop_secs=self.vad_stop_secs,
            audio_bytes=len(audio_data),
            duration_seconds=sample.duration_seconds,
            source_label=source_label or sample.sample_id,
        )

        metrics_observer = MetricsCollectorObserver()
        transcription_observer = TranscriptionCollectorObserver()
        metrics_observer.set_current_sample(sample.sample_id)
        transcription_observer.set_current_sample(sample.sample_id)

        try:
            definition = get_service_definition(service_name.value)
            if definition.needs_aiohttp:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    result = await self._run_pipeline(
                        sample=sample,
                        service_name=service_name,
                        model=model,
                        audio_data=audio_data,
                        metrics_observer=metrics_observer,
                        transcription_observer=transcription_observer,
                        aiohttp_session=session,
                        extra_observers=[debug_observer],
                    )
            else:
                result = await self._run_pipeline(
                    sample=sample,
                    service_name=service_name,
                    model=model,
                    audio_data=audio_data,
                    metrics_observer=metrics_observer,
                    transcription_observer=transcription_observer,
                    extra_observers=[debug_observer],
                )
        except Exception as e:
            logger.error(f"[{service_name.value}] Debug run failed for {sample.sample_id}: {e}")
            result = BenchmarkResult(
                sample_id=sample.sample_id,
                service_name=service_name,
                model_name=model,
                audio_duration_seconds=sample.duration_seconds,
                error=str(e),
            )

        debug_observer.log_done(
            transcription=result.transcription,
            ttfb_seconds=result.ttfb_seconds,
            error=result.error,
        )
        return result

    async def _run_pipeline(
        self,
        sample: AudioSample,
        service_name: ServiceName,
        model: str | None,
        audio_data: bytes,
        metrics_observer: MetricsCollectorObserver,
        transcription_observer: TranscriptionCollectorObserver,
        aiohttp_session: "aiohttp.ClientSession | None" = None,
        extra_observers: list[BaseObserver] | None = None,
    ) -> BenchmarkResult:
        """Run the benchmark pipeline for a single sample.

        Args:
            sample: The audio sample to benchmark.
            service_name: The STT service to use.
            model: Optional model name override.
            audio_data: Raw audio bytes.
            metrics_observer: Observer for collecting TTFB metrics.
            transcription_observer: Observer for collecting transcriptions.
            aiohttp_session: Optional aiohttp session for services that need one.

        Returns:
            BenchmarkResult with TTFB and transcription.
        """
        from stt_benchmark.observers.debug_trace import DebugTraceObserver

        # Create STT service using its factory
        stt_service = create_stt_service(
            service_name,
            aiohttp_session=aiohttp_session,
            grpc_options=self.grpc_options,
        )
        if extra_observers:
            for observer in extra_observers:
                if isinstance(observer, DebugTraceObserver) and hasattr(
                    stt_service, "set_grpc_reply_trace"
                ):
                    stt_service.set_grpc_reply_trace(observer.log_grpc_reply)
                    break

        # Create transport with audio
        # Pass transcription_received event so transport sends silence
        # until transcription arrives (or timeout), then continues for
        # post_transcription_delay to collect additional segments
        stream_ready = getattr(stt_service, "stream_ready_event", None)
        transport = SyntheticInputTransport(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            chunk_ms=self.chunk_ms,
            transcription_received=transcription_observer._transcription_received,
            max_silence_timeout=self.max_silence_timeout_secs,
            post_transcription_delay=self.post_transcription_delay_secs,
            stream_ready=stream_ready,
        )

        vad_processor = VADProcessor(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=self.vad_stop_secs))
        )

        # Build pipeline
        pipeline = Pipeline([transport, vad_processor, stt_service])

        # Create task with observers
        observers: list[BaseObserver] = [
            metrics_observer,
            transcription_observer,
        ]
        if extra_observers:
            observers.extend(extra_observers)

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=self.sample_rate,
                audio_in_channels=1,
                enable_metrics=True,
            ),
            observers=observers,
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
            f"[{service_name.value}] Sample {sample.sample_id}: TTFB={ttfb:.3f}s"
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
