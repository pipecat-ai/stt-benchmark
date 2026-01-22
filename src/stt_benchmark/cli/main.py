"""Main CLI entry point for STT benchmarking."""

import typer

from stt_benchmark.cli.benchmark import app as benchmark_app
from stt_benchmark.cli.download import app as download_app
from stt_benchmark.cli.export import app as export_app
from stt_benchmark.cli.ground_truth import app as ground_truth_app
from stt_benchmark.cli.report import app as report_app
from stt_benchmark.cli.wer import app as wer_app

app = typer.Typer(
    name="stt-benchmark",
    help="STT (Speech-to-Text) benchmarking tool using Pipecat. Measures TTFB and Semantic WER.",
    no_args_is_help=True,
)

# Add subcommands
app.add_typer(download_app, name="download", help="Download and prepare audio samples")
app.add_typer(benchmark_app, name="run", help="Run STT benchmarks")
app.add_typer(ground_truth_app, name="ground-truth", help="Generate ground truth using Gemini")
app.add_typer(wer_app, name="wer", help="Calculate semantic WER metrics")
app.add_typer(report_app, name="report", help="Generate reports and compare services")
app.add_typer(export_app, name="export", help="Export data for a specific service")


@app.callback()
def main():
    """STT Benchmark - Measure TTFB for Speech-to-Text services."""
    pass


if __name__ == "__main__":
    app()
