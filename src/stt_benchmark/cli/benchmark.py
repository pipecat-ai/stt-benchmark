"""CLI command for running STT benchmarks."""

import asyncio
import uuid
from datetime import datetime

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from stt_benchmark.analysis.statistics import compute_statistics, format_statistics_table
from stt_benchmark.config import get_config
from stt_benchmark.evals.runner import run_suite
from stt_benchmark.models import AudioSample, BenchmarkResult, BenchmarkRun, ServiceName
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
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        "-c",
        min=1,
        help="How many samples to benchmark at once, across all services",
    ),
    test: bool = typer.Option(
        False,
        "--test",
        "-t",
        help="Use separate test database (test_results.db) to avoid affecting real data",
    ),
):
    """Run STT TTFB benchmarks on configured services.

    Benchmarks each audio sample with the specified STT services,
    measuring Time To First Byte (TTFB) using Pipecat's MetricsFrame. Each
    sample is played to a bot of its own by Pipecat's eval suite.
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
    console.print(f"Concurrency: {concurrency}")
    if test:
        console.print("[yellow]Test mode: using separate test database[/yellow]")

    async def run():
        # Use test database if --test flag is set
        config = get_config()
        db_path = None
        if test:
            db_path = config.data_dir / "test_results.db"

        db = Database(db_path=db_path)
        await db.initialize()

        # If test mode, copy samples from main database if needed
        if test:
            main_db_path = config.results_db
            copied = await db.copy_samples_from(main_db_path)
            if copied > 0:
                console.print(f"[dim]Copied {copied} samples from main database[/dim]")

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

        # Work out what each service still has to do
        pending_by_service: dict[ServiceName, list[AudioSample]] = {}
        for service_name in service_list:
            if skip_existing:
                pending = await db.get_samples_without_results(service_name, model)
                pending = [s for s in pending if s in samples]
            else:
                pending = samples
            if pending:
                pending_by_service[service_name] = pending
                console.print(f"  {service_name.value}: {len(pending)} pending samples")
            else:
                console.print(f"  {service_name.value}: all samples already benchmarked")

        results: list[BenchmarkResult] = []
        if pending_by_service:
            run_dir = config.data_dir / "eval_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
            console.print(f"\nRun directory: {run_dir}\n")

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                tasks = {
                    service_name: progress.add_task(
                        f"Benchmarking {service_name.value}...", total=len(pending)
                    )
                    for service_name, pending in pending_by_service.items()
                }

                def on_result(result: BenchmarkResult) -> None:
                    progress.advance(tasks[result.service_name])

                results = await run_suite(
                    run_dir,
                    pending_by_service,
                    db,
                    concurrency=concurrency,
                    vad_stop_secs=vad_stop_secs,
                    model=model,
                    on_result=on_result,
                )

        # Compute and display statistics, including results from previous runs
        all_stats = []
        for service_name in service_list:
            ran = [r for r in results if r.service_name == service_name]
            if ran:
                errors = [r for r in ran if r.error]
                console.print(f"\n[bold]{service_name.value}[/bold]")
                console.print(f"  [green]Completed: {len(ran) - len(errors)}[/green]")
                if errors:
                    console.print(f"  [yellow]Errors: {len(errors)}[/yellow]")
                    console.print(f"  [dim]First error: {errors[0].error}[/dim]")

            all_results = await db.get_results_for_service(service_name, model)
            stats = compute_statistics(all_results) if all_results else None
            if stats:
                all_stats.append(stats)
                if ran:
                    console.print(f"  Mean TTFB: {stats.ttfb_mean:.3f}s")
                    console.print(f"  P95 TTFB:  {stats.ttfb_p95:.3f}s")

        # Mark run as complete
        await db.update_run_completed(run_record.run_id)

        # Print summary
        if all_stats:
            console.print("\n")
            console.print(format_statistics_table(all_stats))

        await db.close()

    asyncio.run(run())
