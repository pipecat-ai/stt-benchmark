#!/usr/bin/env python3
"""Generate a Pareto frontier plot of TTFS vs Semantic WER for STT services."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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
                    "wer_mean": wer_summary["wer_mean"] * 100,  # Convert to %
                }

        await db.close()
        return data

    return asyncio.run(fetch())


def plot_pareto_frontier(
    data: dict, output_path: str = "stt_pareto_frontier.png", show: bool = False
):
    """Generate the TTFS vs WER scatter plot with Pareto frontier annotation."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("matplotlib is required for plotting. Install with: uv add matplotlib")
        sys.exit(1)

    # Create figure with extra space at bottom for Pareto annotation
    fig = plt.figure(figsize=(10, 8.5))
    ax = fig.add_axes([0.1, 0.24, 0.85, 0.66])  # [left, bottom, width, height]

    # Plot each service
    ttfb_values = []
    wer_values = []
    names = []

    for name, metrics in data.items():
        ttfb = metrics["ttfb_median"]
        wer = metrics["wer_mean"]
        ttfb_values.append(ttfb)
        wer_values.append(wer)
        names.append(name)

        ax.scatter(ttfb, wer, s=120, zorder=5)
        ax.annotate(
            name,
            (ttfb, wer),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=10,
            fontweight="bold",
        )

    # Configure axes
    ax.set_xlabel("TTFS Median (ms) (lower is better)", fontsize=12)
    ax.set_ylabel("Semantic WER Mean (%) (lower is better)", fontsize=12)
    ax.set_title(
        "STT Pareto Frontier: Latency vs Accuracy",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Set axis limits with padding
    max_ttfb = max(ttfb_values) * 1.15
    max_wer = max(wer_values) * 1.15
    ax.set_xlim(0, max_ttfb)
    ax.set_ylim(0, max_wer)

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
            "Pareto Frontier Services",
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
            service_strs.append(f"{name.capitalize()}: TTFS {ttfb:.0f}ms, WER {wer:.2f}%")

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


def main():
    parser = argparse.ArgumentParser(
        description="Generate Pareto frontier plot of TTFS vs Semantic WER for STT services"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="assets/stt_pareto_frontier.png",
        help="Output file path (default: assets/stt_pareto_frontier.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively",
    )
    args = parser.parse_args()

    print("Fetching data from database...")
    data = get_data_from_db()

    if not data:
        print("No data found. Run benchmarks and WER calculation first.")
        sys.exit(1)

    print(f"Found {len(data)} services with complete metrics")
    for name, metrics in sorted(data.items()):
        print(f"  {name}: TTFS={metrics['ttfb_median']:.0f}ms, WER={metrics['wer_mean']:.2f}%")

    print("\nGenerating plot...")
    plot_pareto_frontier(data, args.output, args.show)


if __name__ == "__main__":
    main()
