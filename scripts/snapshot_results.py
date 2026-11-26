#!/usr/bin/env python3
"""Record the README's results as a dated snapshot, so they can be compared over time.

The README shows the latest results. This copies them into ``results/<label>/``:

    summary.csv            one row per service, the README table's numbers
    run.json               how the results were measured
    stt_pareto_frontier.png, stt_latency_range.png

and rewrites ``results/history.csv``, which holds every snapshot's rows in one
file. Run it after ``stt-benchmark update-readme`` and
``scripts/pareto-frontier-plot.py``; running it again for the same label
replaces that snapshot.

Usage:
    uv run python scripts/snapshot_results.py 2026-11 \\
        --measured-from 2026-11-23 --measured-to 2026-11-26 \\
        --region us-west-2 --instance-type c7i.4xlarge
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from stt_benchmark.evaluation.semantic_wer import (  # noqa: E402
    DEFAULT_JUDGE_EFFORT,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_REPEATS,
    JUDGE_FINGERPRINT,
)
from stt_benchmark.reporting.readme_table import parse_table_rows  # noqa: E402

# summary.csv columns, and the README table key each comes from.
COLUMNS = {
    "vendor": "vendor",
    "model": "model",
    "transcripts_pct": "success_rate",
    "perfect_pct": "perfect_pct",
    "wer_mean_pct": "wer_mean",
    "pooled_wer_pct": "pooled_wer",
    "ttfs_median_ms": "ttfb_median",
    "ttfs_p95_ms": "ttfb_p95",
    "ttfs_p99_ms": "ttfb_p99",
}
PLOTS = ("stt_pareto_frontier.png", "stt_latency_range.png")


def judge_label() -> str:
    """The label ``stt-benchmark wer`` stores with each score from the current judge."""
    sampling = f"effort={DEFAULT_JUDGE_EFFORT}" if DEFAULT_JUDGE_EFFORT else "temperature=0"
    return (
        f"{DEFAULT_JUDGE_MODEL} {sampling} repeats={DEFAULT_JUDGE_REPEATS} rev={JUDGE_FINGERPRINT}"
    )


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_history(results_dir: Path) -> int:
    """Rebuild history.csv from every snapshot's summary. Returns the number of snapshots."""
    summaries = sorted(results_dir.glob("*/summary.csv"))
    with (results_dir / "history.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["snapshot", *COLUMNS])
        writer.writeheader()
        for summary in summaries:
            with summary.open(newline="") as f:
                for row in csv.DictReader(f):
                    writer.writerow({"snapshot": summary.parent.name, **row})
    return len(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("label", help="Snapshot name, such as 2026-11")
    parser.add_argument("--measured-from", required=True, help="First day of measurement (UTC)")
    parser.add_argument("--measured-to", required=True, help="Last day of measurement (UTC)")
    parser.add_argument("--region", required=True, help="Where the benchmark ran from")
    parser.add_argument("--instance-type", help="Machine the benchmark ran on")
    parser.add_argument(
        "--pipecat-version", help="Pipecat version measured with (default: the one installed)"
    )
    parser.add_argument("--samples", type=int, default=1000, help="Samples per service")
    parser.add_argument("--notes", action="append", default=[], help="Caveat to record; repeatable")
    parser.add_argument("--readme", type=Path, default=REPO / "README.md")
    parser.add_argument("--assets", type=Path, default=REPO / "assets")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    args = parser.parse_args()

    rows = parse_table_rows(args.readme.read_text())
    if not rows:
        raise SystemExit(f"No results table found in {args.readme}")

    snapshot = args.results_dir / args.label
    snapshot.mkdir(parents=True, exist_ok=True)

    with (snapshot / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[key] for column, key in COLUMNS.items()})

    run = {
        "snapshot": args.label,
        "measured_from": args.measured_from,
        "measured_to": args.measured_to,
        "region": args.region,
        "instance_type": args.instance_type,
        "samples_per_service": args.samples,
        "pipecat_version": args.pipecat_version or version("pipecat-ai"),
        "benchmark_commit": git_commit(),
        "judge": judge_label(),
        "services": len(rows),
        "notes": args.notes,
    }
    (snapshot / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    for plot in PLOTS:
        if (args.assets / plot).exists():
            shutil.copy(args.assets / plot, snapshot / plot)

    snapshots = write_history(args.results_dir)
    print(f"Wrote {len(rows)} services to {snapshot}")
    print(f"{args.results_dir / 'history.csv'} now covers {snapshots} snapshot(s)")


if __name__ == "__main__":
    main()
