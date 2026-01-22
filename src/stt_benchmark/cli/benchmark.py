"""CLI command for running STT benchmarks."""

import asyncio
import uuid

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from stt_benchmark.analysis.statistics import compute_statistics, format_statistics_table
from stt_benchmark.models import BenchmarkRun, ServiceName
from stt_benchmark.pipeline.benchmark_runner import BenchmarkRunner
from stt_benchmark.services import STT_SERVICES, parse_services_arg
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
def run_benchmark(
    services: str = typer.Option(
        "all",
        "--services",
        "-s",
        help="Comma-separated list of services (deepgram,cartesia,etc.) or 'all'",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-n",
        help="Limit number of samples to benchmark",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name override (service-specific)",
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip samples that already have results",
    ),
    vad_stop_secs: float = typer.Option(
        0.2,
        "--vad-stop-secs",
        "-v",
        help="VAD silence duration to trigger stop (seconds)",
    ),
):
    """Run STT TTFB benchmarks on configured services.

    Benchmarks each audio sample with the specified STT services,
    measuring Time To First Byte (TTFB) using Pipecat's MetricsFrame.
    """
    console.print("\n[bold blue]STT Benchmark - Run Benchmarks[/bold blue]\n")

    # Parse and validate services
    service_list = parse_services(services)

    if not service_list:
        console.print("[red]No services available. Please configure API keys.[/red]")
        console.print("\nSet environment variables:")
        for config in STT_SERVICES.values():
            console.print(f"  {config.api_key_env}")
        raise typer.Exit(1)

    console.print(f"Services: {', '.join(s.value for s in service_list)}")
    if model:
        console.print(f"Model override: {model}")
    if limit:
        console.print(f"Sample limit: {limit}")
    console.print(f"Skip existing: {skip_existing}")
    console.print(f"VAD stop secs: {vad_stop_secs}")

    async def run():
        db = Database()
        await db.initialize()

        # Get samples
        samples = await db.get_all_samples()
        if not samples:
            console.print("\n[red]No samples found. Run 'stt-benchmark download' first.[/red]")
            return

        if limit:
            samples = samples[:limit]

        console.print(f"Samples: {len(samples)}\n")

        # Create benchmark run record
        run_record = BenchmarkRun(
            run_id=str(uuid.uuid4()),
            services=service_list,
            num_samples=len(samples),
        )
        await db.insert_run(run_record)

        # Create benchmark runner
        runner = BenchmarkRunner(vad_stop_secs=vad_stop_secs)

        all_stats = []

        for service_name in service_list:
            console.print(f"\n[bold]Benchmarking {service_name.value}...[/bold]")

            try:
                # Get samples to process
                if skip_existing:
                    pending = await db.get_samples_without_results(service_name, model)
                    pending = [s for s in pending if s in samples]
                else:
                    pending = samples

                if not pending:
                    console.print(f"  All samples already benchmarked for {service_name.value}")
                    # Still compute stats from existing results
                    results = await db.get_results_for_service(service_name, model)
                    if results:
                        stats = compute_statistics(results)
                        if stats:
                            all_stats.append(stats)
                    continue

                console.print(f"  Pending samples: {len(pending)}")

                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        f"Benchmarking {service_name.value}...",
                        total=len(pending),
                    )

                    def callback(current, total, sample_id):
                        progress.update(task, completed=current)

                    results = await runner.benchmark_batch(
                        pending,
                        service_name,
                        model=model,
                        progress_callback=callback,
                    )

                    progress.update(task, completed=len(pending))

                # Save results
                await db.insert_results_batch(results)

                # Compute and display statistics
                successful = len([r for r in results if not r.error])
                errors = len([r for r in results if r.error])

                console.print(f"  [green]Completed: {successful}[/green]")
                if errors > 0:
                    console.print(f"  [yellow]Errors: {errors}[/yellow]")

                # Get all results for this service (including previous runs)
                all_results = await db.get_results_for_service(service_name, model)
                stats = compute_statistics(all_results)
                if stats:
                    all_stats.append(stats)
                    console.print(f"  Mean TTFB: {stats.ttfb_mean:.3f}s")
                    console.print(f"  P95 TTFB:  {stats.ttfb_p95:.3f}s")

            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
                continue

        # Mark run as complete
        await db.update_run_completed(run_record.run_id)

        # Print summary
        if all_stats:
            console.print("\n")
            console.print(format_statistics_table(all_stats))

        await db.close()

    asyncio.run(run())
