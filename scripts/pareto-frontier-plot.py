#!/usr/bin/env python3
"""Generate a Pareto frontier plot of TTFS vs Semantic WER for STT services."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add parent to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

LATENCY_METRICS = {
    "median": {"key": "ttfb_median", "label": "TTFS Median", "suffix": ""},
    "p95": {"key": "ttfb_p95", "label": "TTFS P95", "suffix": "_p95"},
    "p99": {"key": "ttfb_p99", "label": "TTFS P99", "suffix": "_p99"},
}


def get_data_from_db():
    """Fetch service metrics from the database."""
    from stt_benchmark.storage.database import Database

    async def fetch():
        db = Database()
        await db.initialize()

        services = await db.get_services_with_results()
        data = {}

        for service_name, model_name in services:
            transcript_stats = await db.get_service_transcript_stats(service_name, model_name)
            wer_summary = await db.get_service_summary(service_name, model_name)

            if transcript_stats and wer_summary:
                data[service_name.value] = {
                    "ttfb_median": transcript_stats["ttfb_median"] * 1000,  # Convert to ms
                    "ttfb_p95": transcript_stats["ttfb_p95"] * 1000,
                    "ttfb_p99": transcript_stats["ttfb_p99"] * 1000,
                    "pooled_wer": wer_summary["pooled_wer"] * 100,  # Convert to %
                }

        await db.close()
        return data

    return asyncio.run(fetch())


def plot_pareto_frontier(
    data: dict,
    latency_metric: str = "median",
    output_path: str = "stt_pareto_frontier.png",
    show: bool = False,
):
    """Generate the TTFS vs WER scatter plot with Pareto frontier annotation."""
    try:
        import matplotlib.pyplot as plt
        from adjustText import adjust_text
        from matplotlib.patches import FancyBboxPatch
    except ImportError as e:
        if "adjustText" in str(e):
            print("adjustText is required for plotting. Install with: uv add adjustText")
        else:
            print("matplotlib is required for plotting. Install with: uv add matplotlib")
        sys.exit(1)

    metric_info = LATENCY_METRICS[latency_metric]
    ttfb_key = metric_info["key"]
    ttfb_label = metric_info["label"]

    # Create figure with extra space at bottom for Pareto annotation
    fig = plt.figure(figsize=(10, 8.5))
    ax = fig.add_axes([0.1, 0.24, 0.85, 0.66])  # [left, bottom, width, height]

    # Plot each service
    ttfb_values = []
    wer_values = []
    names = []
    texts = []

    for name, metrics in data.items():
        ttfb = metrics[ttfb_key]
        wer = metrics["pooled_wer"]
        ttfb_values.append(ttfb)
        wer_values.append(wer)
        names.append(name)

        ax.scatter(ttfb, wer, s=120, zorder=5)
        texts.append(
            ax.text(
                ttfb,
                wer,
                name,
                fontsize=10,
                fontweight="bold",
            )
        )

    # Configure axes
    ax.set_xlabel(f"{ttfb_label} (ms) (lower is better)", fontsize=12)
    ax.set_ylabel("Semantic WER Pooled (%) (lower is better)", fontsize=12)
    ax.set_title(
        f"STT Pareto Frontier: {ttfb_label} Latency vs Accuracy",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Set axis limits with padding
    max_ttfb = max(ttfb_values) * 1.15
    max_wer = max(wer_values) * 1.15
    ax.set_xlim(0, max_ttfb)
    ax.set_ylim(0, max_wer)

    # Adjust text positions to avoid overlaps
    adjust_text(
        texts,
        x=ttfb_values,
        y=wer_values,
        arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5),
        expand=(1.2, 1.4),
        force_text=(0.5, 1.0),
    )

    # Add reference lines for best values
    best_ttfb = min(ttfb_values)
    best_wer = min(wer_values)
    ax.axhline(y=best_wer, color="green", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=best_ttfb, color="blue", linestyle="--", alpha=0.4, linewidth=1)

    # Add "ideal" corner indicator
    ax.annotate(
        "← ideal",
        (best_ttfb * 0.5, best_wer * 0.5),
        fontsize=10,
        color="gray",
        style="italic",
    )

    # Find Pareto-optimal services (not dominated by any other)
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

    # Sort Pareto optimal by TTFS (fastest first)
    pareto_optimal.sort(key=lambda x: x[1])

    # Draw Pareto frontier line connecting optimal points
    if len(pareto_optimal) > 1:
        frontier_ttfb = [p[1] for p in pareto_optimal]
        frontier_wer = [p[2] for p in pareto_optimal]
        ax.plot(
            frontier_ttfb,
            frontier_wer,
            color="#c44e52",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            zorder=4,
            marker="o",
            markersize=8,
            markerfacecolor="#c44e52",
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

    # Add bottom panel for Pareto frontier services
    if pareto_optimal:
        # Draw background box (positioned below the chart with gap)
        box = FancyBboxPatch(
            (0.02, 0.02),
            0.96,
            0.13,
            boxstyle="round,pad=0.005,rounding_size=0.01",
            facecolor="#f0f7f0",
            edgecolor="#4a7c4a",
            linewidth=1.5,
            transform=fig.transFigure,
            clip_on=False,
        )
        fig.patches.append(box)

        # Header
        fig.text(
            0.04,
            0.13,
            f"Pareto Frontier Services ({ttfb_label})",
            fontsize=11,
            fontweight="bold",
            color="#2d5a2d",
        )

        # Description
        fig.text(
            0.04,
            0.10,
            "These services offer the best trade-off between latency and accuracy "
            "(no other service is better on both metrics):",
            fontsize=9,
            color="#555555",
        )

        # List Pareto optimal services with stats
        service_strs = []
        for name, ttfb, wer in pareto_optimal:
            service_strs.append(f"{name.capitalize()}: {ttfb:.0f}ms, WER {wer:.2f}%")

        # Display services in a wrapped format
        services_text = "    ".join(service_strs)
        fig.text(
            0.04,
            0.05,
            services_text,
            fontsize=10,
            fontweight="medium",
            color="#333333",
        )

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def load_config_file(config_path: str) -> dict:
    """Load plot configuration from a JSON file.

    Supported keys:
        services: list of service names to include (e.g. ["deepgram", "assemblyai"])
        display_names: dict mapping service keys to display labels (e.g. {"deepgram": "Deepgram"})
        latency: latency metric - "median", "p95", "p99", or "all"
        output: output file path or directory
        show: whether to display the plot interactively (true/false)
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate Pareto frontier plot of TTFS vs Semantic WER for STT services"
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
        help="Latency metrics to plot (e.g. -l median p95). Default: median p95",
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
    latency = args.latency or file_config.get("latency", ["median", "p95"])
    services = args.services or file_config.get("services", None)
    display_names = file_config.get("display_names", {})
    show = args.show if args.show is not None else file_config.get("show", False)

    print("Fetching data from database...")
    data = get_data_from_db()

    if not data:
        print("No data found. Run benchmarks and WER calculation first.")
        sys.exit(1)

    # Filter to requested services
    if services:
        data = filter_services(data, services)
        if not data:
            print("No matching services found. Nothing to plot.")
            sys.exit(1)

    # Apply display name mapping
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

    # If output is a directory, generate filenames inside it
    if output_path.is_dir() or output.endswith("/"):
        output_path.mkdir(parents=True, exist_ok=True)
        for metric in metrics_to_plot:
            suffix = LATENCY_METRICS[metric]["suffix"]
            plot_output = output_path / f"{default_basename}{suffix}.png"
            print(f"\nGenerating {LATENCY_METRICS[metric]['label']} plot...")
            plot_pareto_frontier(data, metric, str(plot_output), show)
    else:
        for metric in metrics_to_plot:
            suffix = LATENCY_METRICS[metric]["suffix"]
            if len(metrics_to_plot) > 1:
                plot_output = output_path.parent / f"{output_path.stem}{suffix}{output_path.suffix}"
            else:
                plot_output = output_path
            print(f"\nGenerating {LATENCY_METRICS[metric]['label']} plot...")
            plot_pareto_frontier(data, metric, str(plot_output), show)


if __name__ == "__main__":
    main()
