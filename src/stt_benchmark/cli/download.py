"""CLI command for downloading and preparing audio samples."""

import asyncio

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from stt_benchmark.config import get_config
from stt_benchmark.dataset.downloader import DatasetDownloader

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
def download(
    num_samples: int = typer.Option(
        100,
        "--num-samples",
        "-n",
        help="Number of audio samples to download",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducible sample selection",
    ),
    offset: int = typer.Option(
        0,
        "--offset",
        "-o",
        help="Number of samples to skip (for incremental downloads)",
    ),
):
    """Download and prepare audio samples from HuggingFace dataset.

    Downloads samples from the pipecat-ai/smart-turn-data-v3.1-train dataset,
    converts them to 16kHz PCM audio, and stores metadata in SQLite.
    """
    config = get_config()

    console.print("\n[bold blue]STT Benchmark - Download Dataset[/bold blue]\n")
    console.print(f"Dataset: {config.dataset_name}")
    console.print(f"Samples: {num_samples}")
    console.print(f"Seed: {seed}")
    console.print(f"Offset: {offset}")
    console.print(f"Output: {config.audio_dir}\n")

    async def run():
        downloader = DatasetDownloader(
            num_samples=num_samples,
            seed=seed,
            offset=offset,
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=4)

            def callback(current, total, message):
                progress.update(task, completed=current, description=message)

            samples = await downloader.download_and_prepare(progress_callback=callback)

            progress.update(task, completed=4, description="Complete!")

        console.print(f"\n[green]✓ Downloaded {len(samples)} samples[/green]")

        # Show duration distribution
        if samples:
            durations = [s.duration_seconds for s in samples]
            short = len([d for d in durations if d < 2])
            medium = len([d for d in durations if 2 <= d < 5])
            long = len([d for d in durations if 5 <= d < 10])
            very_long = len([d for d in durations if d >= 10])

            console.print("\n[bold]Duration Distribution:[/bold]")
            console.print(f"  0-2s:   {short:4d} samples")
            console.print(f"  2-5s:   {medium:4d} samples")
            console.print(f"  5-10s:  {long:4d} samples")
            console.print(f"  10s+:   {very_long:4d} samples")

    asyncio.run(run())
