"""The benchmark's eval bot: one STT service behind a VAD, audio in, measurements out.

The eval suite spawns this bot once per sample, so every sample gets a fresh
connection to the STT service. The harness plays the sample, marks its end with
a DTMF keypress, and keeps sending silence. The bot collects the service's
final transcripts, its TTFS and any error it reports, decides when the sample
is done (see :mod:`stt_benchmark.evals.collector`), and reports the result to
the harness in an RTVI server message.

The runner body (the manifest entry's ``runner_body.data``) names the service::

    service: deepgram        # a key of stt_benchmark.services.STT_SERVICES
    vad_stop_secs: 0.2       # optional
"""

import asyncio
from collections.abc import Awaitable, Callable

import aiohttp
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.evals.transport import EvalTransportParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport
from pipecat.workers.runner import WorkerRunner

from stt_benchmark.config import get_config
from stt_benchmark.evals.collector import (
    SAMPLE_KEY,
    AudioEndObserver,
    SampleResult,
    STTErrorObserver,
    collect_sample,
)
from stt_benchmark.observers.metrics_collector import MetricsCollectorObserver
from stt_benchmark.observers.transcription_collector import TranscriptionCollectorObserver
from stt_benchmark.services import create_stt_service, get_service_definition, parse_service_name

transport_params = {
    "eval": lambda: EvalTransportParams(audio_in_enabled=True),
}


def build_worker(
    audio_in: FrameProcessor,
    stt: FrameProcessor,
    audio_out: FrameProcessor,
    *,
    vad_stop_secs: float,
    idle_timeout_secs: float | None = None,
) -> tuple[PipelineWorker, Callable[[], Awaitable[SampleResult]]]:
    """Build the bot's pipeline, and what reports its sample.

    Args:
        audio_in: Where the sample's audio enters the pipeline.
        stt: The STT service being benchmarked.
        audio_out: Where the bot's messages leave the pipeline.
        vad_stop_secs: The VAD's stop silence.
        idle_timeout_secs: How long the pipeline may stay idle.

    Returns:
        The worker, and a coroutine function that waits for the sample to be
        done, sends its result as a server message, and returns it.
    """
    config = get_config()
    vad_processor = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=vad_stop_secs))
    )

    audio_end = AudioEndObserver()
    errors = STTErrorObserver()
    metrics = MetricsCollectorObserver()
    transcripts = TranscriptionCollectorObserver()
    metrics.set_current_sample(SAMPLE_KEY)
    transcripts.set_current_sample(SAMPLE_KEY)

    worker = PipelineWorker(
        Pipeline([audio_in, vad_processor, stt, audio_out]),
        params=PipelineParams(
            audio_in_sample_rate=config.sample_rate,
            audio_in_channels=1,
            enable_metrics=True,
        ),
        observers=[audio_end, errors, metrics, transcripts],
        idle_timeout_secs=idle_timeout_secs,
    )

    async def report_sample() -> SampleResult:
        result = await collect_sample(
            audio_end,
            transcripts,
            metrics,
            errors=errors,
            max_silence_secs=config.max_silence_timeout_secs,
            transcription_timeout_secs=config.transcription_timeout_secs,
        )
        logger.info(f"Sample done: {result}")
        await worker.queue_frame(RTVIServerMessageFrame(data=result.as_message()))
        return result

    return worker, report_sample


async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    aiohttp_session: aiohttp.ClientSession | None,
):
    body = runner_args.body or {}
    service_name = parse_service_name(body["service"])
    vad_stop_secs = float(body.get("vad_stop_secs", get_config().vad_stop_secs))
    logger.info(f"Starting STT benchmark bot: {service_name.value} (VAD stop {vad_stop_secs}s)")

    worker, report_sample = build_worker(
        transport.input(),
        create_stt_service(service_name, aiohttp_session=aiohttp_session),
        transport.output(),
        vad_stop_secs=vad_stop_secs,
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    collecting: asyncio.Task | None = None

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal collecting
        collecting = asyncio.create_task(report_sample())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        if collecting is not None:
            collecting.cancel()
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""
    body = runner_args.body
    if not body or "service" not in body:
        raise ValueError("this bot needs a runner body naming the STT service (see module doc)")
    definition = get_service_definition(parse_service_name(body["service"]).value)

    transport = await create_transport(runner_args, transport_params)
    if definition.needs_aiohttp:
        async with aiohttp.ClientSession() as session:
            await run_bot(transport, runner_args, session)
    else:
        await run_bot(transport, runner_args, None)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
