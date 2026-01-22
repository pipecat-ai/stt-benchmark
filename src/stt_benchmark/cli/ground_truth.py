"""CLI command for generating ground truth transcriptions."""

import asyncio

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from stt_benchmark.config import get_config
from stt_benchmark.ground_truth.gemini_transcriber import GeminiTranscriber
from stt_benchmark.storage.database import Database

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
def generate_ground_truth(
    ctx: typer.Context,
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-n",
        help="Limit number of samples to transcribe",
    ),
    model: str = typer.Option(
        "gemini-3-flash-preview",
        "--model",
        "-m",
        help="Gemini model to use for transcription",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-transcribe samples that already have ground truth",
    ),
):
    """Generate ground truth transcriptions using Gemini.

    Uses Google's Gemini model to transcribe audio samples,
    creating reference transcriptions for WER calculation.

    Requires GOOGLE_API_KEY environment variable.

    Subcommands:
      iterate  - Run a repeatable transcription iteration (saves to JSONL)
      review   - Interactive review of transcription runs with audio playback
      list     - List available transcription runs
    """
    # If a subcommand was invoked, don't run the default behavior
    if ctx.invoked_subcommand is not None:
        return

    config = get_config()

    if not config.google_api_key:
        console.print("[red]Error: GOOGLE_API_KEY not set[/red]")
        console.print("\nSet the environment variable:")
        console.print("  export GOOGLE_API_KEY='your-api-key'")
        raise typer.Exit(1)

    console.print("\n[bold blue]STT Benchmark - Generate Ground Truth[/bold blue]\n")
    console.print(f"Model: {model}")
    if limit:
        console.print(f"Limit: {limit}")
    console.print(f"Force re-transcribe: {force}")

    async def run():
        db = Database()
        await db.initialize()

        # Get samples
        if force:
            samples = await db.get_all_samples()
        else:
            samples = await db.get_samples_without_ground_truth()

        if not samples:
            console.print("\n[green]All samples already have ground truth![/green]")
            return

        if limit:
            samples = samples[:limit]

        console.print(f"Samples to transcribe: {len(samples)}\n")

        # Create transcriber
        transcriber = GeminiTranscriber(model_name=model)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating ground truth...", total=len(samples))

            def callback(current, total, sample_id):
                progress.update(task, completed=current)

            results = await transcriber.transcribe_batch(
                samples,
                progress_callback=callback,
                save_incrementally=True,
                force=force,
            )

            progress.update(task, completed=len(samples))

        # Summary
        console.print(f"\n[green]✓ Generated {len(results)} ground truth transcriptions[/green]")

        # Show stats
        gt_count = await db.get_ground_truth_count()
        sample_count = await db.get_sample_count()
        console.print(f"\nGround truth coverage: {gt_count}/{sample_count} samples")

        await db.close()

    asyncio.run(run())


@app.command("iterate")
def iterate_command(
    samples: int = typer.Option(
        100,
        "--samples",
        "-n",
        help="Number of samples to transcribe",
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        "-c",
        help="Clear existing ground truths before running",
    ),
):
    """Run a repeatable transcription iteration.

    Generates transcriptions for a fixed set of samples and saves to JSONL
    for longitudinal comparison as you iterate on the transcription prompt.

    The output is saved to stt_benchmark_data/ground_truth_runs/<timestamp>.jsonl
    """
    config = get_config()

    if not config.google_api_key:
        console.print("[red]Error: GOOGLE_API_KEY not set[/red]")
        raise typer.Exit(1)

    console.print("\n[bold blue]STT Benchmark - Ground Truth Iteration[/bold blue]\n")
    console.print(f"Samples: {samples}")
    console.print(f"Clear existing: {clear}")

    async def run():
        from stt_benchmark.ground_truth.run_iteration import run_iteration

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Transcribing...", total=samples)

            def callback(current, total, sample_id):
                progress.update(task, completed=current)

            output_path = await run_iteration(
                num_samples=samples,
                clear_existing=clear,
                progress_callback=callback,
            )

            progress.update(task, completed=samples)

        console.print("\n[green]✓ Iteration complete[/green]")
        console.print(f"Output: {output_path}")
        console.print("\nTo review this run:")
        console.print(f"  stt-benchmark ground-truth review {output_path.stem}")

    asyncio.run(run())


@app.command("list")
def list_runs_command():
    """List available transcription runs.

    Shows all ground truth transcription runs that can be reviewed.
    """
    from stt_benchmark.ground_truth.run_iteration import list_runs

    console.print("\n[bold blue]STT Benchmark - Ground Truth Runs[/bold blue]\n")

    runs = list_runs()

    if not runs:
        console.print("[yellow]No runs found.[/yellow]")
        console.print("\nCreate a run with:")
        console.print("  stt-benchmark ground-truth iterate --samples 100")
        return

    table = Table(title="Available Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Model", style="dim")
    table.add_column("Samples", justify="right")
    table.add_column("Reviewed", justify="right")

    for run in runs:
        reviewed = f"{run['reviewed_count']}/{run['num_samples']}"
        table.add_row(
            run["run_id"],
            run["model"],
            str(run["num_samples"]),
            reviewed,
        )

    console.print(table)
    console.print("\nTo review a run:")
    console.print("  stt-benchmark ground-truth review <run_id>")


@app.command("review")
def review_command(
    run_id: str = typer.Argument(
        ...,
        help="Run ID to review (from 'list' command)",
    ),
):
    """Interactive review of transcription runs with audio playback.

    Listen to audio samples and approve/note/edit transcriptions.
    Requires ffplay (from ffmpeg) for audio playback.

    Controls:
      [p] Play audio
      [r] Replay audio
      [a] Approve transcription
      [e] Edit transcription (saves correction to database)
      [n] Add note (flag for review)
      [Enter] Skip to next
      [q] Quit
    """
    from stt_benchmark.ground_truth.evaluate_run import get_run_path, run_evaluation

    run_path = get_run_path(run_id)

    if not run_path:
        console.print(f"[red]Run not found: {run_id}[/red]")
        console.print("\nUse 'stt-benchmark ground-truth list' to see available runs")
        raise typer.Exit(1)

    # Check for ffplay
    import shutil

    if not shutil.which("ffplay"):
        console.print("[yellow]Warning: ffplay not found[/yellow]")
        console.print("Audio playback requires ffplay (from ffmpeg)")
        console.print("Install with: brew install ffmpeg")
        console.print()

    # Run the interactive evaluation
    run_evaluation(run_path)


@app.command("edit")
def edit_command(
    sample_id: str = typer.Argument(
        ...,
        help="Sample ID to edit (e.g., 'sample_0001')",
    ),
    text: str | None = typer.Option(
        None,
        "--text",
        "-t",
        help="New transcription text. If not provided, opens interactive editor.",
    ),
):
    """Manually edit ground truth for a specific sample.

    Use this to correct transcription errors without going through
    the full review workflow.

    Examples:
      stt-benchmark ground-truth edit sample_0001 --text "corrected text here"
      stt-benchmark ground-truth edit sample_0001  # interactive
    """
    from datetime import datetime, timezone

    from stt_benchmark.models import GroundTruth
    from stt_benchmark.storage.database import Database

    async def run():
        db = Database()
        await db.initialize()

        # Check if sample exists
        sample = await db.get_sample(sample_id)
        if not sample:
            console.print(f"[red]Sample not found: {sample_id}[/red]")
            return

        # Get current ground truth
        current_gt = await db.get_ground_truth(sample_id)

        if current_gt:
            console.print("\n[bold]Current ground truth:[/bold]")
            console.print(f'"{current_gt.text}"')
            console.print(f"[dim]Model: {current_gt.model_used}[/dim]\n")
        else:
            console.print(f"\n[yellow]No ground truth exists for {sample_id}[/yellow]\n")

        # Get new text
        if text:
            new_text = text
        else:
            console.print("Enter new transcription (or press Enter to cancel):")
            new_text = input("> ").strip()

        if not new_text:
            console.print("[yellow]Cancelled[/yellow]")
            return

        # Save to database
        gt = GroundTruth(
            sample_id=sample_id,
            text=new_text,
            model_used="human_corrected",
            generated_at=datetime.now(timezone.utc),
        )
        await db.insert_ground_truth(gt)
        await db.close()

        console.print(f"\n[green]✓ Updated ground truth for {sample_id}[/green]")
        console.print(f'New text: "{new_text[:60]}{"..." if len(new_text) > 60 else ""}"')

    asyncio.run(run())
