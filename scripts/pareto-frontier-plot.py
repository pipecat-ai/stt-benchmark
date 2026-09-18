#!/usr/bin/env python3
"""Generate a Pareto frontier plot of TTFS vs Semantic WER for STT services."""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stt_benchmark.reporting.readme_table import METRIC_KEYS, parse_table_rows  # noqa: E402

LATENCY_METRICS = {
    "median": {"key": "ttfb_median", "label": "TTFS Median", "suffix": ""},
    "p95": {"key": "ttfb_p95", "label": "TTFS P95", "suffix": "_p95"},
    "p99": {"key": "ttfb_p99", "label": "TTFS P99", "suffix": "_p99"},
}

# Chart chrome: quiet ink/grid tokens, one hue for all data points, and a
# single highlight color reserved for the Pareto frontier band.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
DOT = "#2a78d6"
BAND = "#1baf7a"

# Latency range chart: one blue hue stepping lighter as the percentile
# deepens, with the connector drawn as a gradient between the same
# endpoints (pre-blended toward the surface so it sits under the dots).
RANGE_MEDIAN = "#1c5cab"
RANGE_P95 = "#3987e5"
RANGE_P99 = "#86b6ef"

# Hand-placed label positions for regions too dense for automatic layout:
# label -> (dx, dy, horizontal alignment), offsets in points from the dot.
# Keyed by latency metric; labels not listed fall back to automatic
# placement (adjustText). Override per-metric via the "label_offsets"
# config key.
DEFAULT_LABEL_OFFSETS = {
    "median": {
        "NVIDIA Nemotron 3.0 ASR (en)": (8, 6, "left"),
        "Deepgram": (8, 2, "left"),
        "Soniox stt-rt-v4": (-9, 4, "right"),
        "Soniox stt-rt-v5": (-9, -12, "right"),
        "AssemblyAI universal-3-5-pro": (8, 19, "left"),
        "Cartesia ink-2": (8, -3, "left"),
        "Speechmatics": (10, -2, "left"),
        "Speechmatics linden-1": (10, -9, "left"),
        "AssemblyAI universal-3-6-pro": (-10, -12, "right"),
        "Meta": (10, -4, "left"),
    },
}


def find_pareto_optimal(names: list, ttfb_values: list, wer_values: list) -> list:
    """Return (name, ttfb, wer) tuples not dominated by any other service, fastest first."""
    pareto_optimal = []
    for i, (name, ttfb, wer) in enumerate(zip(names, ttfb_values, wer_values, strict=False)):
        is_dominated = False
        for j, (other_ttfb, other_wer) in enumerate(zip(ttfb_values, wer_values, strict=False)):
            if i != j and other_ttfb <= ttfb and other_wer <= wer:
                if other_ttfb < ttfb or other_wer < wer:
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_optimal.append((name, ttfb, wer))

    pareto_optimal.sort(key=lambda x: x[1])
    return pareto_optimal


def plot_pareto_frontier(
    data: dict,
    latency_metric: str = "median",
    output_path: str = "stt_pareto_frontier.png",
    show: bool = False,
    label_offsets: dict | None = None,
):
    """Generate the TTFS vs WER scatter plot with the Pareto frontier highlighted."""
    try:
        import matplotlib.pyplot as plt
        from adjustText import adjust_text
        from matplotlib.lines import Line2D
    except ImportError as e:
        if "adjustText" in str(e):
            print("adjustText is required for plotting. Install with: uv add adjustText")
        else:
            print("matplotlib is required for plotting. Install with: uv add matplotlib")
        sys.exit(1)

    metric_info = LATENCY_METRICS[latency_metric]
    ttfb_key = metric_info["key"]
    ttfb_label = metric_info["label"]
    if label_offsets is None:
        label_offsets = DEFAULT_LABEL_OFFSETS.get(latency_metric, {})

    names = list(data)
    ttfb_values = [data[n][ttfb_key] for n in names]
    wer_values = [data[n]["pooled_wer"] for n in names]
    pareto_optimal = find_pareto_optimal(names, ttfb_values, wer_values)

    fig, ax = plt.subplots(figsize=(12.5, 8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Pareto frontier: a soft highlight band through the optimal points,
    # plus a halo ring on each one.
    frontier_ttfb = [p[1] for p in pareto_optimal]
    frontier_wer = [p[2] for p in pareto_optimal]
    if len(pareto_optimal) > 1:
        ax.plot(
            frontier_ttfb,
            frontier_wer,
            color=BAND,
            alpha=0.22,
            linewidth=30,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=1,
        )
    ax.scatter(
        frontier_ttfb,
        frontier_wer,
        s=200,
        facecolors="none",
        edgecolors=BAND,
        linewidths=1.8,
        alpha=0.85,
        zorder=4,
    )

    # All services: one hue, ring in the surface color for separation.
    ax.scatter(
        ttfb_values, wer_values, s=55, color=DOT, edgecolors=SURFACE, linewidths=1.2, zorder=5
    )

    # Labels: hand-placed offsets where the layout is too dense for the
    # automatic solver, adjustText everywhere else.
    leader_line = {"arrowstyle": "-", "color": MUTED, "alpha": 0.6, "lw": 0.8}
    texts = []
    for name, ttfb, wer in zip(names, ttfb_values, wer_values, strict=False):
        if name in label_offsets:
            dx, dy, ha = label_offsets[name]
            ax.annotate(
                name,
                (ttfb, wer),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=7.5,
                color=INK_2,
                zorder=6,
                arrowprops={**leader_line, "shrinkA": 1, "shrinkB": 5},
            )
        else:
            texts.append(ax.text(ttfb, wer, name, fontsize=7.5, color=INK_2, zorder=6))

    # Axis limits: pad around the data instead of forcing a zero origin,
    # so dense regions keep their resolution.
    x_range = max(ttfb_values) - min(ttfb_values) or max(ttfb_values) * 0.2 or 1
    y_range = max(wer_values) - min(wer_values) or max(wer_values) * 0.2 or 1
    ax.set_xlim(min(ttfb_values) - 0.08 * x_range, max(ttfb_values) + 0.10 * x_range)
    ax.set_ylim(min(wer_values) - 0.08 * y_range, max(wer_values) + 0.12 * y_range)

    # Recessive chrome: hairline grid, no top/right spines, muted ticks.
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=10)

    ax.set_xlabel(f"{ttfb_label} (ms) (lower is better)", fontsize=11, color=INK)
    ax.set_ylabel("Semantic WER Pooled (%) (lower is better)", fontsize=11, color=INK)
    ax.set_title(
        f"STT Pareto Frontier: {ttfb_label} Latency vs Accuracy",
        fontsize=13,
        fontweight="bold",
        color=INK,
        pad=14,
    )

    if texts:
        adjust_text(
            texts,
            x=ttfb_values,
            y=wer_values,
            ax=ax,
            arrowprops=leader_line,
            expand=(1.5, 1.9),
            force_text=(0.6, 1.2),
        )

    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=BAND,
                alpha=0.35,
                linewidth=12,
                solid_capstyle="round",
                label="Pareto frontier",
            )
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor=GRID,
        fontsize=11,
    )

    plt.savefig(output_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"Plot saved to: {output_path}")
    if pareto_optimal:
        frontier_strs = [
            f"{name}: {ttfb:.0f}ms, WER {wer:.2f}%" for name, ttfb, wer in pareto_optimal
        ]
        print(f"Pareto frontier: {' | '.join(frontier_strs)}")

    if show:
        plt.show()

    plt.close(fig)


def _mix(hex_color: str, other: str, t: float) -> tuple:
    """Blend hex_color toward other by t (0..1), returning an RGB tuple."""
    a = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(other[i : i + 2], 16) for i in (1, 3, 5)]
    return tuple((av + (bv - av) * t) / 255 for av, bv in zip(a, b, strict=True))


def plot_latency_range(
    data: dict,
    output_path: str = "stt_latency_range.png",
    show: bool = False,
):
    """Generate the per-service latency distribution chart (median / P95 / P99).

    One row per service sorted by median (fastest on top), with a gradient
    connector from median out to P99 — the length of each row is the size
    of the service's latency tail.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("matplotlib is required for plotting. Install with: uv add matplotlib")
        sys.exit(1)

    gradient = LinearSegmentedColormap.from_list(
        "latency", [_mix(RANGE_MEDIAN, SURFACE, 0.55), _mix(RANGE_P99, SURFACE, 0.55)]
    )
    services = sorted(data, key=lambda n: data[n]["ttfb_median"])

    fig, ax = plt.subplots(figsize=(12.5, 8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ys = range(len(services), 0, -1)  # fastest at the top
    max_p99 = 0.0
    for y, name in zip(ys, services, strict=True):
        med = data[name]["ttfb_median"]
        p95 = data[name]["ttfb_p95"]
        p99 = data[name]["ttfb_p99"]
        max_p99 = max(max_p99, p99)
        if p99 > med:
            gx = np.linspace(med, p99, 60)
            segs = [((gx[i], y), (gx[i + 1], y)) for i in range(len(gx) - 1)]
            ax.add_collection(
                LineCollection(
                    segs,
                    colors=gradient(np.linspace(0, 1, len(segs))),
                    linewidths=3,
                    capstyle="butt",
                    zorder=2,
                )
            )
        ax.scatter([p99], [y], s=55, color=RANGE_P99, edgecolors=SURFACE, linewidths=1.4, zorder=4)
        ax.scatter([p95], [y], s=55, color=RANGE_P95, edgecolors=SURFACE, linewidths=1.4, zorder=5)
        ax.scatter(
            [med], [y], s=70, color=RANGE_MEDIAN, edgecolors=SURFACE, linewidths=1.4, zorder=6
        )

    ax.set_yticks(list(ys))
    ax.set_yticklabels(services, fontsize=9, color=INK_2)
    ax.set_ylim(0.3, len(services) + 0.7)

    # Zero origin on purpose: rows never collide, and it keeps the row
    # lengths (tail sizes) honestly comparable.
    ax.set_xlim(0, max_p99 * 1.05)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=10)
    ax.tick_params(axis="y", length=0)

    ax.set_xlabel("TTFS (ms) (lower is better)", fontsize=11, color=INK)
    ax.set_title(
        "STT Latency Distribution: Median / P95 / P99 by Service",
        fontsize=13,
        fontweight="bold",
        color=INK,
        pad=14,
    )

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=c,
            markeredgecolor=SURFACE,
            label=lab,
        )
        for c, lab in [(RANGE_MEDIAN, "Median"), (RANGE_P95, "P95"), (RANGE_P99, "P99")]
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor=GRID,
        fontsize=10,
    )

    plt.savefig(output_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"Plot saved to: {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def load_config_file(config_path: str) -> dict:
    """Load plot configuration from a JSON file.

    Supported keys:
        services: list of service names to include (e.g. ["deepgram", "assemblyai"])
        display_names: optional dict of per-service label overrides. Labels are
            derived from the registry (vendor / model_label) by default; only add
            entries here to override a derived label.
        charts: chart types to generate - ["pareto", "range"] (default both)
        latency: latency metric for the Pareto charts - "median", "p95", "p99"
        output: output file path or directory
        show: whether to display the plot interactively (true/false)
        label_offsets: optional per-metric hand-placed label positions,
            e.g. {"median": {"Deepgram": [8, 2, "left"]}} with offsets in
            points from the dot. Overrides the script's built-in defaults
            for that metric; labels not listed use automatic placement.
    """
    path = Path(config_path)
    if not path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(path) as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in config file: {e}")
            sys.exit(1)

    return config


def apply_display_names(data: dict, display_names: dict[str, str]) -> dict:
    """Remap data keys using a display name mapping.

    Matching is case-insensitive on the keys. Any service without
    a display name entry keeps its original key.
    """
    name_map = {k.lower(): v for k, v in display_names.items()}
    return {name_map.get(k.lower(), k): v for k, v in data.items()}


def filter_services(data: dict, service_names: list[str]) -> dict:
    """Filter data to only include the specified services.

    Matching is case-insensitive. Exits with an error if any requested
    service is not found in the data.
    """
    # Build a case-insensitive lookup from available data
    available = {k.lower(): k for k in data}
    filtered = {}
    missing = []

    for name in service_names:
        key = name.lower()
        if key in available:
            original_key = available[key]
            filtered[original_key] = data[original_key]
        else:
            missing.append(name)

    if missing:
        print(f"Warning: services not found in data: {', '.join(missing)}")
        print(f"Available services: {', '.join(sorted(data.keys()))}")

    return filtered


def get_data_from_readme(readme_path: Path) -> dict:
    """Build plot data from the README "Results Summary" table.

    The table between the RESULTS_TABLE markers is the single source of truth, so
    the rendered Pareto charts always match the published numbers. Returns a dict
    keyed by a display label, with values in ms / %.

    Point labels are the vendor name when a vendor has a single row, or
    ``"{vendor} {model}"`` when a vendor ships multiple rows (so they don't
    collide on the chart).
    """
    from collections import Counter

    if not readme_path.exists():
        print(f"README not found: {readme_path}")
        sys.exit(1)

    rows = parse_table_rows(readme_path.read_text())

    vendor_counts = Counter(r["vendor"] for r in rows)
    data = {}
    for r in rows:
        if vendor_counts[r["vendor"]] > 1 and r["model"] and r["model"].upper() != "N/A":
            label = f"{r['vendor']} {r['model']}"
        else:
            label = r["vendor"]
        data[label] = {k: r[k] for k in METRIC_KEYS}
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate latency/accuracy charts for STT services: "
        "Pareto frontier plots of TTFS vs Semantic WER, and a per-service "
        "latency distribution (median/P95/P99) chart"
    )
    parser.add_argument(
        "--charts",
        nargs="+",
        choices=["pareto", "range"],
        default=None,
        help="Chart types to generate (default: pareto range)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path or directory (default: assets/)",
    )
    parser.add_argument(
        "-l",
        "--latency",
        nargs="+",
        choices=["median", "p95", "p99"],
        default=None,
        help="Latency metrics for the Pareto charts (e.g. -l median p95). Default: median",
    )
    parser.add_argument(
        "-s",
        "--services",
        nargs="+",
        default=None,
        help="Services to include in the plot (e.g. -s deepgram assemblyai groq)",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to a JSON config file with plot settings",
    )
    parser.add_argument(
        "--readme",
        default="README.md",
        metavar="README_PATH",
        help="Path to the README whose results table is the data source "
        "(default: README.md, resolved relative to the repo root). The README is "
        "the single source of truth; this script never writes to it.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=None,
        help="Display the plot interactively",
    )
    args = parser.parse_args()

    # Load config file if provided (CLI args take precedence)
    file_config = {}
    if args.config:
        file_config = load_config_file(args.config)
        print(f"Loaded config from: {args.config}")

    # Resolve settings: CLI args > config file > defaults
    output = args.output or file_config.get("output", "assets/")
    charts = args.charts or file_config.get("charts", ["pareto", "range"])
    latency = args.latency or file_config.get("latency", ["median"])
    services = args.services or file_config.get("services", None)
    display_names = file_config.get("display_names", {})
    label_offsets_cfg = file_config.get("label_offsets", {})
    show = args.show if args.show is not None else file_config.get("show", False)

    # The README results table is the single source of truth for the published
    # numbers; plots are always rendered from it (never from the DB), so this
    # script never modifies the README. To add/update a row, use
    # `stt-benchmark update-readme`.
    readme_path = Path(args.readme)
    if not readme_path.is_absolute() and not readme_path.exists():
        readme_path = Path(__file__).parent.parent / args.readme
    print(f"Reading metrics from README table: {readme_path}")
    data = get_data_from_readme(readme_path)

    if not data:
        print("No data rows parsed from the README table. Nothing to plot.")
        sys.exit(1)

    # Filter to requested services (case-insensitive match on the label)
    if services:
        data = filter_services(data, services)
        if not data:
            print("No matching services found. Nothing to plot.")
            sys.exit(1)

    # Optional per-label display-name overrides from the config file.
    if display_names:
        data = apply_display_names(data, display_names)

    print(f"Found {len(data)} services with complete metrics")
    for name, metrics in sorted(data.items()):
        print(
            f"  {name}: Median={metrics['ttfb_median']:.0f}ms, "
            f"P95={metrics['ttfb_p95']:.0f}ms, "
            f"P99={metrics['ttfb_p99']:.0f}ms, "
            f"WER={metrics['pooled_wer']:.2f}%"
        )

    # Determine which metrics to plot
    if isinstance(latency, str):
        metrics_to_plot = [latency]
    else:
        metrics_to_plot = list(latency)

    valid_metrics = set(LATENCY_METRICS.keys())
    invalid = [m for m in metrics_to_plot if m not in valid_metrics]
    if invalid:
        print(f"Invalid latency metrics: {', '.join(invalid)}")
        print(f"Valid options: {', '.join(sorted(valid_metrics))}")
        sys.exit(1)

    # Generate plots
    output_path = Path(output)
    default_basename = "stt_pareto_frontier"
    is_dir_output = output_path.is_dir() or output.endswith("/")

    if "pareto" in charts:
        # If output is a directory, generate filenames inside it
        if is_dir_output:
            output_path.mkdir(parents=True, exist_ok=True)
            for metric in metrics_to_plot:
                suffix = LATENCY_METRICS[metric]["suffix"]
                plot_output = output_path / f"{default_basename}{suffix}.png"
                print(f"\nGenerating {LATENCY_METRICS[metric]['label']} plot...")
                plot_pareto_frontier(
                    data, metric, str(plot_output), show, label_offsets_cfg.get(metric)
                )
        else:
            for metric in metrics_to_plot:
                suffix = LATENCY_METRICS[metric]["suffix"]
                if len(metrics_to_plot) > 1:
                    plot_output = (
                        output_path.parent / f"{output_path.stem}{suffix}{output_path.suffix}"
                    )
                else:
                    plot_output = output_path
                print(f"\nGenerating {LATENCY_METRICS[metric]['label']} plot...")
                plot_pareto_frontier(
                    data, metric, str(plot_output), show, label_offsets_cfg.get(metric)
                )

    if "range" in charts:
        # The range chart has a fixed basename; with a file-path output it
        # lands next to that file, otherwise inside the output directory.
        if is_dir_output:
            output_path.mkdir(parents=True, exist_ok=True)
            range_output = output_path / "stt_latency_range.png"
        else:
            range_output = output_path.parent / "stt_latency_range.png"
        print("\nGenerating latency distribution plot...")
        plot_latency_range(data, str(range_output), show)


if __name__ == "__main__":
    main()
