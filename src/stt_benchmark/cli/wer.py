"""CLI command for calculating semantic WER metrics using Claude."""

import asyncio
import statistics

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from stt_benchmark.config import get_config
from stt_benchmark.evaluation.semantic_wer import SemanticWEREvaluator
from stt_benchmark.models import ServiceName
from stt_benchmark.services import parse_services_arg
from stt_benchmark.storage.database import Database

app = typer.Typer()
console = Console()


def parse_services(services: str) -> list[ServiceName]:
    """Parse service names from comma-separated string."""
    try:
        return parse_services_arg(services)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None


@app.callback(invoke_without_command=True)
def calculate_wer(
    services: str = typer.Option(
        "all",
        "--services",
        "-s",
        help="Comma-separated list of services or 'all'",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name filter",
    ),
    force_recalculate: bool = typer.Option(
        False,
        "--force-recalculate",
        "-f",
        help="Force recalculation of WER even for samples that already have metrics",
    ),
    test: bool = typer.Option(
        False,
        "--test",
        "-t",
        help="Use test database (test_results.db) instead of main database",
    ),
):
    """Calculate semantic WER metrics for transcription results.

    Uses Claude to evaluate transcription quality, counting only errors
    that would impact how an LLM agent understands the user's intent.

    This is NOT traditional WER - it's semantic WER that ignores:
    - Punctuation/formatting differences
    - Singular/plural variations ("license" vs "licenses")
    - Contractions ("I'm" vs "I am")
    - Filler words ("um", "uh")

    And only counts errors that change meaning for an LLM agent.

    Requires:
    - ANTHROPIC_API_KEY set in environment
    - Ground truth generated (stt-benchmark ground-truth)
    """
    console.print("\n[bold blue]STT Benchmark - Calculate Semantic WER[/bold blue]\n")

    if test:
        console.print("[yellow]Using test database[/yellow]\n")

    # Check for Anthropic API key
    config = get_config()
    if not config.anthropic_api_key:
        console.print("[red]ANTHROPIC_API_KEY not set![/red]")
        console.print("Set ANTHROPIC_API_KEY in your .env file or environment.")
        raise typer.Exit(1)

    # Determine database path
    db_path = None
    if test:
        db_path = config.data_dir / "test_results.db"

    service_list = parse_services(services)

    if not service_list:
        console.print("[yellow]No services with results found.[/yellow]")
        return

    console.print(f"Services: {', '.join(s.value for s in service_list)}")
    if model:
        console.print(f"Model filter: {model}")
    if force_recalculate:
        console.print("[yellow]Force recalculate: ON[/yellow]")

    console.print("\n[dim]Using Claude for semantic WER evaluation...[/dim]")
    console.print("[dim]This evaluates errors that would impact LLM agent understanding.[/dim]\n")

    async def run():
        db = Database(db_path=db_path)
        await db.initialize()

        # Check ground truth coverage
        gt_count = await db.get_ground_truth_count()
        sample_count = await db.get_sample_count()

        if gt_count == 0:
            console.print("\n[red]No ground truth found![/red]")
            if db_path:
                console.print(
                    "Run 'stt-benchmark ground-truth import <file> --test' to import ground truth to test database."
                )
            else:
                console.print(
                    "Run 'stt-benchmark ground-truth' first to generate reference transcriptions."
                )
            return

        console.print(f"Ground truth coverage: {gt_count}/{sample_count} samples\n")

        evaluator = SemanticWEREvaluator(db_path=db_path)
        all_stats = []

        for service_name in service_list:
            console.print(f"\n[bold]Evaluating semantic WER for {service_name.value}...[/bold]")

            # Delete existing WER metrics if force recalculate
            if force_recalculate:
                await db.delete_wer_metrics_for_service(service_name, model)
                await db.delete_semantic_wer_traces_for_service(service_name, model)
                console.print("  [yellow]Deleted existing WER metrics and traces[/yellow]")

            # Get samples that need WER calculation
            pending = await db.get_samples_without_wer(service_name, model)

            if not pending:
                console.print("  All samples already have WER metrics")
                # Still show existing stats
                metrics = await db.get_wer_metrics_for_service(service_name, model)
                if metrics:
                    stats = compute_wer_stats(service_name, metrics)
                    all_stats.append(stats)
                    console.print(f"  Mean WER: {stats['wer_mean']:.2%}")
                continue

            console.print(f"  Samples to evaluate: {len(pending)}")

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                progress_task = progress.add_task("Evaluating semantic WER...", total=len(pending))

                def callback(current, total, sample_id, task_id=progress_task):
                    progress.update(task_id, completed=current)

                metrics = await evaluator.evaluate_service(
                    service_name,
                    model_name=model,
                    progress_callback=callback,
                )

                progress.update(progress_task, completed=len(pending))

            # Get all metrics for stats
            all_metrics = await db.get_wer_metrics_for_service(service_name, model)
            if all_metrics:
                stats = compute_wer_stats(service_name, all_metrics)
                all_stats.append(stats)
                console.print(f"  [green]Completed: {len(metrics)} samples[/green]")
                console.print(f"  Mean Semantic WER: {stats['wer_mean']:.2%}")

        # Print summary table
        if all_stats:
            console.print("\n")
            print_wer_summary(all_stats)

        await evaluator.close()
        await db.close()

    asyncio.run(run())


def compute_wer_stats(service_name: ServiceName, metrics: list) -> dict:
    """Compute aggregate semantic WER statistics."""
    wer_values = [m.wer for m in metrics if m.wer < float("inf")]

    # Compute pooled WER
    total_errors = sum(m.substitutions + m.deletions + m.insertions for m in metrics)
    total_ref_words = sum(m.reference_words for m in metrics)
    pooled_wer = total_errors / total_ref_words if total_ref_words > 0 else 0.0

    return {
        "service_name": service_name,
        "num_samples": len(metrics),
        "wer_mean": statistics.mean(wer_values) if wer_values else 0.0,
        "wer_median": statistics.median(wer_values) if wer_values else 0.0,
        "wer_std": statistics.stdev(wer_values) if len(wer_values) > 1 else 0.0,
        "wer_min": min(wer_values) if wer_values else 0.0,
        "wer_max": max(wer_values) if wer_values else 0.0,
        "pooled_wer": pooled_wer,
    }


def print_wer_summary(stats_list: list[dict]):
    """Print semantic WER summary table."""
    table = Table(title="Semantic WER Summary")

    table.add_column("Service", style="cyan", no_wrap=True)
    table.add_column("Samples", justify="right")
    table.add_column("WER Mean", justify="right")
    table.add_column("WER Median", justify="right")
    table.add_column("WER Min", justify="right")
    table.add_column("WER Max", justify="right")
    table.add_column("Pooled WER", justify="right")

    # Sort by mean WER
    sorted_stats = sorted(stats_list, key=lambda x: x["wer_mean"])

    for stats in sorted_stats:
        table.add_row(
            stats["service_name"].value,
            str(stats["num_samples"]),
            f"{stats['wer_mean']:.2%}",
            f"{stats['wer_median']:.2%}",
            f"{stats['wer_min']:.2%}",
            f"{stats['wer_max']:.2%}",
            f"{stats['pooled_wer']:.2%}",
        )

    console.print(table)

    # Rankings
    console.print("\n[bold]Rankings (by mean semantic WER, lower is better):[/bold]")
    for i, stats in enumerate(sorted_stats, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        console.print(f"  {medal} {stats['service_name'].value}: {stats['wer_mean']:.2%}")
