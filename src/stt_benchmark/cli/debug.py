"""CLI command for debugging a single sample through the benchmark pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from stt_benchmark.debug.audio_source import list_samples, load_audio_file, resolve_sample
from stt_benchmark.models import AudioSample, ServiceName
from stt_benchmark.pipeline.benchmark_runner import BenchmarkRunner
from stt_benchmark.services import GrpcServiceOptions, parse_service_name

app = typer.Typer()
console = Console()


def _parse_single_service(services: str) -> ServiceName:
    if services.strip().lower() == "all" or "," in services:
        raise typer.BadParameter(
            "Debug requires exactly one service (no commas, not 'all')."
        )
    try:
        return parse_service_name(services)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None


@app.callback(invoke_without_command=True)
def debug_run(
    services: str | None = typer.Option(
        None,
        "--services",
        "-s",
        help="Single STT service to debug (e.g. asr_backend, speech_proxy)",
    ),
    file: Path | None = typer.Option(
        None,
        "--file",
        "-f",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to input audio (.pcm, .wav, or .mp3)",
    ),
    sample_id: str | None = typer.Option(
        None,
        "--sample-id",
        help="Benchmark sample ID from results database",
    ),
    sample_index: int | None = typer.Option(
        None,
        "--sample-index",
        min=0,
        help="Benchmark sample index (0-based) from results database",
    ),
    list_samples_flag: bool = typer.Option(
        False,
        "--list-samples",
        help="List benchmark audio samples and exit",
    ),
    test: bool = typer.Option(
        False,
        "--test",
        "-t",
        help="Use test database (test_results.db) for --sample-id / --sample-index",
    ),
    vad_stop_secs: float = typer.Option(
        0.2,
        "--vad-stop-secs",
        "-v",
        help="VAD silence duration to trigger stop (seconds)",
    ),
    chunk_ms: int = typer.Option(
        20,
        "--chunk-ms",
        help="Input audio chunk duration in milliseconds",
    ),
    sample_rate: int = typer.Option(
        16000,
        "--sample-rate",
        min=8000,
        help="Target PCM sample rate in Hz; audio is resampled before recognition",
    ),
    asr_backend_url: str = typer.Option(
        "localhost:50052",
        "--asr-backend-url",
        help="gRPC URL for asr_backend / asr_backend_exteou",
    ),
    asr_backend_use_ssl: bool = typer.Option(
        False,
        "--asr-backend-use-ssl/--no-asr-backend-use-ssl",
        help="Use TLS for asr_backend connection",
    ),
    language: str = typer.Option(
        "en",
        "--language",
        help="Language ID for asr_backend services",
    ),
    speech_proxy_url: str = typer.Option(
        "speech-proxy.main.stage.aiphoria.pro:443",
        "--speech-proxy-url",
        help="gRPC URL for speech_proxy",
    ),
    speech_proxy_use_ssl: bool = typer.Option(
        True,
        "--speech-proxy-use-ssl/--no-speech-proxy-use-ssl",
        help="Use TLS for speech_proxy connection",
    ),
    recognizer: str = typer.Option(
        "asr_deepgram_en_nova3",
        "--recognizer",
        help="Recognizer name for speech_proxy",
    ),
) -> None:
    """Debug one audio sample through the benchmark pipeline with VAD + ASR tracing."""
    if list_samples_flag:
        asyncio.run(list_samples(test))
        return

    if not services:
        raise typer.BadParameter("--services is required unless using --list-samples.")

    source_count = sum(
        1 for value in (file, sample_id, sample_index) if value is not None
    )
    if source_count != 1:
        raise typer.BadParameter(
            "Provide exactly one audio source: --file, --sample-id, or --sample-index."
        )

    service_name = _parse_single_service(services)

    async def run() -> None:
        grpc_options = GrpcServiceOptions(
            asr_backend_url=asr_backend_url,
            asr_backend_use_ssl=asr_backend_use_ssl,
            asr_backend_language=language,
            speech_proxy_url=speech_proxy_url,
            speech_proxy_use_ssl=speech_proxy_use_ssl,
            speech_proxy_recognizer=recognizer,
            sample_rate=sample_rate,
        )
        runner = BenchmarkRunner(
            vad_stop_secs=vad_stop_secs,
            chunk_ms=chunk_ms,
            sample_rate=sample_rate,
            grpc_options=grpc_options,
        )

        audio_data: bytes | None = None
        source_label: str
        if file is not None:
            audio_data, duration_seconds = load_audio_file(file, sample_rate=sample_rate)
            sample = AudioSample(
                sample_id="debug",
                audio_path=str(file),
                duration_seconds=duration_seconds,
                dataset_index=-1,
            )
            source_label = str(file)
        elif sample_id is not None:
            sample = await resolve_sample(sample_id=sample_id, sample_index=None, test=test)
            source_label = sample.sample_id
        else:
            sample = await resolve_sample(
                sample_id=None, sample_index=sample_index, test=test
            )
            source_label = sample.sample_id

        result = await runner.debug_sample(
            sample,
            service_name,
            source_label=source_label,
            audio_data=audio_data,
        )

        if result.error:
            console.print(f"[red]Error: {result.error}[/red]")
            raise typer.Exit(1)

        ttfb_text = (
            f"{result.ttfb_seconds * 1000:.0f}ms"
            if result.ttfb_seconds is not None
            else "N/A"
        )
        console.print(
            f"\n[bold green]Summary:[/bold green] service={service_name.value} "
            f"ttfb={ttfb_text} transcription={result.transcription!r}"
        )

    try:
        asyncio.run(run())
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
