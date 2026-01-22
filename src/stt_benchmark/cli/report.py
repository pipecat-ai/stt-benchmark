"""CLI command for generating benchmark reports."""

import asyncio
import statistics
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from stt_benchmark.config import get_config
from stt_benchmark.models import ServiceName
from stt_benchmark.services import parse_service_name
from stt_benchmark.storage.database import Database

app = typer.Typer()
console = Console()


def parse_service(service: str) -> ServiceName:
    """Parse a single service name."""
    try:
        return parse_service_name(service)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None


@app.callback(invoke_without_command=True)
def report(
    service: str | None = typer.Option(
        None,
        "--service",
        "-s",
        help="Service to generate detailed report for. If not specified, shows summary of all services.",
    ),
    output_dir: str = typer.Option(
        "stt_benchmark_data",
        "--output",
        "-o",
        help="Output directory for report files",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name filter",
    ),
    errors: int | None = typer.Option(
        None,
        "--errors",
        "-e",
        help="Show N worst samples (highest WER) for the specified service",
    ),
    test: bool = typer.Option(
        False,
        "--test",
        "-t",
        help="Use test database (test_results.db) instead of main database",
    ),
):
    """Generate benchmark reports.

    By default, shows a summary table comparing all benchmarked services.

    Use --service to generate detailed validation files for a specific service:
    - validation_summary.txt: Overview with statistics and outliers
    - validation_full.csv: Complete data for all samples

    Use --errors N with --service to show the N worst samples in the terminal.

    Note: This uses semantic WER which only counts errors that would
    impact how an LLM agent understands the user's intent.
    """
    console.print("\n[bold blue]STT Benchmark - Report[/bold blue]\n")

    if test:
        console.print("[yellow]Using test database[/yellow]\n")

    # If --errors is specified, --service is required
    if errors is not None and service is None:
        console.print("[red]--errors requires --service to be specified[/red]")
        raise typer.Exit(1)

    # Determine database path
    db_path = None
    if test:
        config = get_config()
        db_path = config.data_dir / "test_results.db"

    if service is None:
        # Default: show summary of all services
        asyncio.run(_show_all_services_summary(db_path))
    elif errors is not None:
        # Show worst samples for a service
        service_name = parse_service(service)
        asyncio.run(_show_worst_samples(service_name, model, errors, db_path))
    else:
        # Generate detailed report for a single service
        service_name = parse_service(service)
        output_path = Path(output_dir)
        asyncio.run(_generate_detailed_report(service_name, model, output_path, db_path))


async def _show_all_services_summary(db_path: Path | None = None):
    """Show summary table of all benchmarked services."""
    db = Database(db_path=db_path)
    await db.initialize()

    # Get all services with results (not just WER)
    services_with_results = await db.get_services_with_results()
    services_with_wer = await db.get_services_with_wer_metrics()
    services_with_wer_set = set(services_with_wer)

    if not services_with_results:
        console.print("[yellow]No benchmark results found.[/yellow]")
        console.print("Run 'stt-benchmark run --services <service>' first.")
        await db.close()
        return

    # Get sample and ground truth counts
    sample_count = await db.get_sample_count()
    gt_count = await db.get_ground_truth_count()

    console.print(f"Total samples: {sample_count}")
    console.print(f"Ground truth available: {gt_count}\n")

    # Check if any services have WER
    has_any_wer = len(services_with_wer) > 0

    # Build summary table (matches README format)
    table = Table(title="Service Comparison")
    table.add_column("Service", style="cyan", no_wrap=True)
    table.add_column("Transcripts", justify="right")
    if has_any_wer:
        table.add_column("Perfect", justify="right")
        table.add_column("WER Mean", justify="right")
        table.add_column("Pooled WER", justify="right")
    table.add_column("TTFS Median", justify="right")
    table.add_column("TTFS P95", justify="right")
    table.add_column("TTFS P99", justify="right")

    summaries = []
    for service_name, model_name in services_with_results:
        transcript_stats = await db.get_service_transcript_stats(service_name, model_name)

        # Get WER summary if available
        wer_summary = None
        if (service_name, model_name) in services_with_wer_set:
            wer_summary = await db.get_service_summary(service_name, model_name)

        if transcript_stats:
            summaries.append((service_name, model_name, transcript_stats, wer_summary))

            # Format transcripts as m/n (x%)
            transcripts_pct = transcript_stats["success_rate"] * 100
            transcripts_str = f"{transcript_stats['successful_transcripts']}/{transcript_stats['total_runs']} ({transcripts_pct:.1f}%)"

            row_data = [
                service_name.value,
                transcripts_str,
            ]

            if has_any_wer:
                if wer_summary:
                    # Perfect: percentage of samples with 0% WER
                    perfect_pct = (
                        wer_summary["perfect_count"] / wer_summary["sample_count"] * 100
                        if wer_summary["sample_count"] > 0
                        else 0
                    )

                    row_data.extend(
                        [
                            f"{perfect_pct:.1f}%",
                            f"{wer_summary['wer_mean'] * 100:.2f}%",
                            f"{wer_summary['pooled_wer'] * 100:.2f}%",
                        ]
                    )
                else:
                    row_data.extend(["-", "-", "-"])

            row_data.extend(
                [
                    f"{transcript_stats['ttfb_median'] * 1000:.0f}ms",
                    f"{transcript_stats['ttfb_p95'] * 1000:.0f}ms",
                    f"{transcript_stats['ttfb_p99'] * 1000:.0f}ms",
                ]
            )

            table.add_row(*row_data)

    await db.close()

    console.print(table)

    # Show rankings
    if summaries:
        # WER rankings (only if we have WER data)
        summaries_with_wer = [(s, m, ts, ws) for s, m, ts, ws in summaries if ws is not None]
        if summaries_with_wer:
            console.print("\n[bold]Rankings (by Perfect %):[/bold]")
            ranked_perfect = sorted(
                summaries_with_wer,
                key=lambda x: x[3]["perfect_count"] / x[3]["sample_count"]
                if x[3]["sample_count"] > 0
                else 0,
                reverse=True,
            )
            for i, (service, model, _, wer_summary) in enumerate(ranked_perfect, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                name = f"{service.value}" + (f" ({model})" if model else "")
                perfect_pct = (
                    wer_summary["perfect_count"] / wer_summary["sample_count"] * 100
                    if wer_summary["sample_count"] > 0
                    else 0
                )
                console.print(f"  {medal} {name}: {perfect_pct:.1f}%")

            console.print("\n[bold]Rankings (by Pooled WER):[/bold]")
            ranked_wer = sorted(summaries_with_wer, key=lambda x: x[3]["pooled_wer"])
            for i, (service, model, _, wer_summary) in enumerate(ranked_wer, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                name = f"{service.value}" + (f" ({model})" if model else "")
                console.print(f"  {medal} {name}: {wer_summary['pooled_wer'] * 100:.2f}%")

        # TTFS rankings
        console.print("\n[bold]Rankings (by TTFS Median):[/bold]")
        ranked_ttfs = sorted(summaries, key=lambda x: x[2]["ttfb_median"])
        for i, (service, model, transcript_stats, _) in enumerate(ranked_ttfs, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            name = f"{service.value}" + (f" ({model})" if model else "")
            console.print(f"  {medal} {name}: {transcript_stats['ttfb_median'] * 1000:.0f}ms")

    if not services_with_wer:
        console.print("\n[dim]Run 'stt-benchmark wer' to calculate semantic WER metrics.[/dim]")
    console.print("[dim]Use --service <name> to generate detailed reports for a service.[/dim]")


async def _show_worst_samples(
    service_name: ServiceName, model_name: str | None, limit: int, db_path: Path | None = None
):
    """Show the worst N samples for a service."""
    db = Database(db_path=db_path)
    await db.initialize()

    report_data = await db.get_report_data(service_name, model_name)

    if not report_data:
        console.print(f"[yellow]No data found for {service_name.value}[/yellow]")
        await db.close()
        return

    await db.close()

    # Data is already sorted by WER descending
    worst = report_data[:limit]

    table = Table(title=f"Top {limit} Worst Samples - {service_name.value}")
    table.add_column("Sample ID", style="cyan", max_width=20)
    table.add_column("WER", justify="right")
    table.add_column("S/D/I", justify="right")
    table.add_column("TTFB", justify="right")
    table.add_column("Duration", justify="right")

    for r in worst:
        ttfb_ms = r["ttfb"] * 1000 if r["ttfb"] else 0
        table.add_row(
            r["sample_id"][:20],
            f"{r['wer'] * 100:.1f}%",
            f"{r['substitutions']}/{r['deletions']}/{r['insertions']}",
            f"{ttfb_ms:.0f}ms",
            f"{r['duration']:.1f}s",
        )

    console.print(table)

    # Show details for worst samples
    console.print("\n[bold]Details:[/bold]")
    for r in worst[:5]:  # Show details for top 5
        console.print(f"\n[cyan]{r['sample_id']}[/cyan] (WER: {r['wer'] * 100:.1f}%)")
        console.print(f"  [dim]Ground Truth:[/dim] {r['ground_truth']}")
        console.print(f"  [dim]Transcription:[/dim] {r['transcription']}")
        if r.get("normalized_reference"):
            console.print(f"  [dim]Normalized Ref:[/dim] {r['normalized_reference']}")
        if r.get("normalized_hypothesis"):
            console.print(f"  [dim]Normalized Hyp:[/dim] {r['normalized_hypothesis']}")


async def _generate_detailed_report(
    service_name: ServiceName,
    model_name: str | None,
    output_path: Path,
    db_path: Path | None = None,
):
    """Generate detailed report files for a single service."""
    db = Database(db_path=db_path)
    await db.initialize()

    # Check if we have data
    wer_count = await db.get_wer_metrics_count(service_name, model_name)
    if wer_count == 0:
        console.print(f"[red]No WER metrics found for {service_name.value}[/red]")
        console.print("Run 'stt-benchmark wer' first to calculate metrics.")
        await db.close()
        return

    output_path.mkdir(parents=True, exist_ok=True)

    console.print(f"Service: {service_name.value}")
    console.print(f"Samples with WER: {wer_count}")
    console.print(f"Output directory: {output_path}\n")

    # Get transcript success stats
    transcript_stats = await db.get_service_transcript_stats(service_name, model_name)

    # Get all data for report
    report_data = await db.get_report_data(service_name, model_name)

    if not report_data:
        console.print("[red]No data found for report[/red]")
        await db.close()
        return

    # Calculate WER statistics
    wer_values = [r["wer"] for r in report_data]
    perfect_count = sum(1 for w in wer_values if w == 0)
    outlier_count = sum(1 for w in wer_values if w > 0.5)

    sorted_wer = sorted(wer_values)
    median_wer = sorted_wer[len(sorted_wer) // 2] if sorted_wer else 0

    # Calculate TTFB statistics
    ttfb_values = [r["ttfb"] for r in report_data if r["ttfb"] is not None]
    if ttfb_values:
        ttfb_mean = statistics.mean(ttfb_values)
        ttfb_median = statistics.median(ttfb_values)
        ttfb_min = min(ttfb_values)
        ttfb_max = max(ttfb_values)
        ttfb_p95 = (
            sorted(ttfb_values)[int(len(ttfb_values) * 0.95)]
            if len(ttfb_values) > 1
            else ttfb_values[0]
        )
    else:
        ttfb_mean = ttfb_median = ttfb_min = ttfb_max = ttfb_p95 = 0.0

    # Distribution buckets
    buckets = {
        "0% (perfect)": sum(1 for w in wer_values if w == 0),
        "1-5%": sum(1 for w in wer_values if 0 < w <= 0.05),
        "6-10%": sum(1 for w in wer_values if 0.05 < w <= 0.10),
        "11-20%": sum(1 for w in wer_values if 0.10 < w <= 0.20),
        "21-50%": sum(1 for w in wer_values if 0.20 < w <= 0.50),
        "50%+ (outliers)": sum(1 for w in wer_values if w > 0.50),
    }

    # Generate summary text file
    summary_path = output_path / "validation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"STT BENCHMARK VALIDATION REPORT - {service_name.value.upper()}\n")
        f.write("=" * 80 + "\n\n")

        f.write("NOTE: This uses Semantic WER - only counts errors that would\n")
        f.write("impact how an LLM agent understands the user's intent.\n\n")

        f.write("--- OVERALL STATISTICS ---\n")
        f.write(f"Total samples: {len(report_data)}\n\n")

        if transcript_stats:
            f.write("Transcription Rate:\n")
            f.write(f"  Total runs: {transcript_stats['total_runs']}\n")
            f.write(f"  Transcribed: {transcript_stats['successful_transcripts']}\n")
            f.write(f"  Failed: {transcript_stats['failed_transcripts']}\n")
            f.write(f"  % Transcribed: {transcript_stats['success_rate'] * 100:.1f}%\n\n")

        f.write("Semantic WER (Word Error Rate):\n")
        f.write(f"  Mean: {sum(wer_values) / len(wer_values) * 100:.2f}%\n")
        f.write(f"  Median: {median_wer * 100:.2f}%\n")
        f.write(f"  Min: {min(wer_values) * 100:.2f}%\n")
        f.write(f"  Max: {max(wer_values) * 100:.2f}%\n")
        f.write(f"  Perfect matches: {perfect_count}\n")
        f.write(f"  Outliers (>50%): {outlier_count}\n\n")

        f.write("TTFB (Time To First Byte):\n")
        f.write(f"  Mean: {ttfb_mean * 1000:.0f}ms\n")
        f.write(f"  Median: {ttfb_median * 1000:.0f}ms\n")
        f.write(f"  Min: {ttfb_min * 1000:.0f}ms\n")
        f.write(f"  Max: {ttfb_max * 1000:.0f}ms\n")
        f.write(f"  P95: {ttfb_p95 * 1000:.0f}ms\n")
        f.write(f"  Samples with TTFB: {len(ttfb_values)}\n\n")

        f.write("--- WER DISTRIBUTION ---\n")
        for bucket, count in buckets.items():
            f.write(f"  {bucket}: {count}\n")
        f.write("\n")

        # Outliers section
        outliers = [r for r in report_data if r["wer"] > 0.20]
        if outliers:
            f.write("--- OUTLIERS (Semantic WER > 20%) ---\n")
            for r in sorted(outliers, key=lambda x: -x["wer"]):
                f.write(f"\nSample: {r['sample_id']}\n")
                f.write(f"  WER: {r['wer'] * 100:.1f}%\n")
                f.write(
                    f"  Errors: S={r['substitutions']} D={r['deletions']} I={r['insertions']}\n"
                )
                ttfb_ms = r["ttfb"] * 1000 if r["ttfb"] else 0
                f.write(f"  TTFB: {ttfb_ms:.0f}ms\n")
                f.write(f"  Audio duration: {r['duration']:.2f}s\n")
                f.write(f"  Ground Truth: {r['ground_truth']}\n")
                f.write(f"  Transcription: {r['transcription']}\n")
                if r.get("normalized_reference"):
                    f.write(f"  Normalized Ref: {r['normalized_reference']}\n")
                if r.get("normalized_hypothesis"):
                    f.write(f"  Normalized Hyp: {r['normalized_hypothesis']}\n")
        else:
            f.write("--- NO OUTLIERS (Semantic WER > 20%) ---\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Generated files:\n")
        f.write(f"  - {summary_path.name}: This summary\n")
        f.write("  - validation_full.csv: Complete data for all samples\n")
        f.write("=" * 80 + "\n")

    # Generate CSV file
    csv_path = output_path / "validation_full.csv"
    with open(csv_path, "w") as f:
        # Header
        f.write(
            "sample_id,audio_secs,ttfb_ms,WER_pct,substitutions,deletions,"
            "insertions,ref_words,gt_chars,transcription_chars,ground_truth,transcription\n"
        )

        # Sort by WER descending
        for r in sorted(report_data, key=lambda x: -x["wer"]):
            # Escape quotes in text fields
            gt = r["ground_truth"].replace('"', '""')
            tr = r["transcription"].replace('"', '""')
            ttfb_ms = r["ttfb"] * 1000 if r["ttfb"] else 0

            f.write(
                f"{r['sample_id']},"
                f"{r['duration']:.3f},"
                f"{ttfb_ms:.1f},"
                f"{r['wer'] * 100:.2f},"
                f"{r['substitutions']},"
                f"{r['deletions']},"
                f"{r['insertions']},"
                f"{r['ref_words']},"
                f"{len(r['ground_truth'])},"
                f"{len(r['transcription'])},"
                f'"{gt}",'
                f'"{tr}"\n'
            )

    await db.close()

    # Print summary to console
    console.print("[green]✓ Reports generated:[/green]")
    console.print(f"  • {summary_path}")
    console.print(f"  • {csv_path}")
    console.print()

    # Show summary table
    table = Table(title=f"Semantic WER Summary - {service_name.value}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Samples", str(len(report_data)))
    table.add_row("", "")  # Spacer
    if transcript_stats:
        table.add_row("[bold]Transcription Rate[/bold]", "")
        table.add_row("  Total Runs", str(transcript_stats["total_runs"]))
        table.add_row("  Transcribed", str(transcript_stats["successful_transcripts"]))
        table.add_row("  Failed", str(transcript_stats["failed_transcripts"]))
        table.add_row("  % Transcribed", f"{transcript_stats['success_rate'] * 100:.1f}%")
        table.add_row("", "")  # Spacer
    table.add_row("[bold]Semantic WER[/bold]", "")
    table.add_row("  Mean", f"{sum(wer_values) / len(wer_values) * 100:.2f}%")
    table.add_row("  Median", f"{median_wer * 100:.2f}%")
    table.add_row("  Perfect Matches", str(perfect_count))
    table.add_row("  Outliers (>50%)", str(outlier_count))
    table.add_row("", "")  # Spacer
    table.add_row("[bold]TTFB[/bold]", "")
    table.add_row("  Mean", f"{ttfb_mean * 1000:.0f}ms")
    table.add_row("  Median", f"{ttfb_median * 1000:.0f}ms")
    table.add_row("  P95", f"{ttfb_p95 * 1000:.0f}ms")

    console.print(table)

    # Distribution table
    dist_table = Table(title="Semantic WER Distribution")
    dist_table.add_column("Range", style="cyan")
    dist_table.add_column("Count", justify="right")
    dist_table.add_column("Percentage", justify="right")

    for bucket, count in buckets.items():
        pct = count / len(report_data) * 100 if report_data else 0
        dist_table.add_row(bucket, str(count), f"{pct:.1f}%")

    console.print(dist_table)

    # Show outliers if any
    if outliers:
        console.print(f"\n[yellow]⚠ {len(outliers)} outliers found (Semantic WER > 20%)[/yellow]")
        console.print("See validation_summary.txt for details.")
