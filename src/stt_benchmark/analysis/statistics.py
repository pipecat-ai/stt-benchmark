"""Statistics computation for benchmark results."""

import statistics

from stt_benchmark.models import AggregateStatistics, BenchmarkResult, ServiceName


def compute_statistics(
    results: list[BenchmarkResult],
    service_name: ServiceName | None = None,
    model_name: str | None = None,
) -> AggregateStatistics | None:
    """Compute aggregate statistics from benchmark results.

    Args:
        results: List of benchmark results.
        service_name: Filter by service name (optional, infers from results).
        model_name: Filter by model name (optional).

    Returns:
        AggregateStatistics or None if no valid results.
    """
    if not results:
        return None

    # Filter results if needed
    filtered = results
    if service_name:
        filtered = [r for r in filtered if r.service_name == service_name]
    if model_name:
        filtered = [r for r in filtered if r.model_name == model_name]

    if not filtered:
        return None

    # Infer service name from first result
    svc = service_name or filtered[0].service_name
    mdl = model_name or filtered[0].model_name

    # Separate successful results (with TTFB) from errors
    successful = [r for r in filtered if r.ttfb_seconds is not None and r.error is None]
    errors = [r for r in filtered if r.error is not None]

    # Non-error utterances with no measurable TTFB -> dropped from the latency
    # percentiles. Two distinct sub-modes:
    #   * early final BEFORE end-of-speech WITH content (truncated transcript)
    #     -> a real false-positive endpoint. This is the FPR numerator.
    #   * no usable post-anchor final captured (empty transcript) -> a no-final
    #     defect (often a recognizer/harness reply-contract mismatch), tracked
    #     separately so it is not read as premature endpointing.
    # FPR is over non-error utterances (valid + dropped).
    no_ttfb = [r for r in filtered if r.error is None and r.ttfb_seconds is None]
    no_final = [r for r in no_ttfb if not (r.transcription or "").strip()]
    early_final = [r for r in no_ttfb if (r.transcription or "").strip()]
    num_non_error = len(filtered) - len(errors)
    fpr = (len(early_final) / num_non_error) if num_non_error else None

    if not successful:
        return AggregateStatistics(
            service_name=svc,
            model_name=mdl,
            num_samples=len(filtered),
            num_errors=len(errors),
            num_premature_eos=len(no_ttfb),
            num_no_final=len(no_final),
            fpr=fpr,
        )

    # Extract TTFB values
    ttfb_values = [r.ttfb_seconds for r in successful]

    # Compute basic statistics
    stats = AggregateStatistics(
        service_name=svc,
        model_name=mdl,
        num_samples=len(filtered),
        num_errors=len(errors),
        num_premature_eos=len(no_ttfb),
        num_no_final=len(no_final),
        fpr=fpr,
        ttfb_mean=statistics.mean(ttfb_values),
        ttfb_median=statistics.median(ttfb_values),
        ttfb_std=statistics.stdev(ttfb_values) if len(ttfb_values) > 1 else 0.0,
        ttfb_min=min(ttfb_values),
        ttfb_max=max(ttfb_values),
        ttfb_p50=_percentile(ttfb_values, 50),
        ttfb_p90=_percentile(ttfb_values, 90),
        ttfb_p95=_percentile(ttfb_values, 95),
        ttfb_p99=_percentile(ttfb_values, 99),
    )

    # Compute TTFB by audio duration bucket
    stats.ttfb_by_duration = _compute_ttfb_by_duration(successful)

    return stats


def _percentile(values: list[float], p: int) -> float:
    """Compute the p-th percentile of values.

    Args:
        values: List of values.
        p: Percentile (0-100).

    Returns:
        The p-th percentile value.
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    # Calculate the index
    k = (p / 100) * (n - 1)
    f = int(k)
    c = f + 1 if f + 1 < n else f

    # Linear interpolation
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def _compute_ttfb_by_duration(results: list[BenchmarkResult]) -> dict[str, float]:
    """Compute mean TTFB by audio duration bucket.

    Args:
        results: List of successful benchmark results.

    Returns:
        Dictionary mapping bucket names to mean TTFB values.
    """
    buckets = {
        "0-2s": [],
        "2-5s": [],
        "5-10s": [],
        "10s+": [],
    }

    for result in results:
        duration = result.audio_duration_seconds
        ttfb = result.ttfb_seconds

        if duration < 2:
            buckets["0-2s"].append(ttfb)
        elif duration < 5:
            buckets["2-5s"].append(ttfb)
        elif duration < 10:
            buckets["5-10s"].append(ttfb)
        else:
            buckets["10s+"].append(ttfb)

    # Compute means for non-empty buckets
    return {
        bucket: statistics.mean(values) if values else None
        for bucket, values in buckets.items()
        if values  # Only include non-empty buckets
    }


def format_statistics_table(stats_list: list[AggregateStatistics]) -> str:
    """Format statistics as a text table.

    Args:
        stats_list: List of AggregateStatistics objects.

    Returns:
        Formatted table string.
    """
    if not stats_list:
        return "No statistics available."

    lines = []
    lines.append("=" * 80)
    lines.append("STT TTFB Benchmark Results")
    lines.append("=" * 80)

    for stats in stats_list:
        lines.append("")
        model_str = f" ({stats.model_name})" if stats.model_name else ""
        lines.append(f"Service: {stats.service_name.value}{model_str}")
        lines.append("-" * 40)
        lines.append(f"  Samples: {stats.num_samples} ({stats.num_errors} errors)")
        num_non_error = stats.num_samples - stats.num_errors
        num_valid = num_non_error - stats.num_premature_eos
        num_early_final = stats.num_premature_eos - stats.num_no_final
        lines.append(f"  Valid (measurable TTFB): {num_valid}/{num_non_error}")
        if stats.fpr is not None and num_non_error:
            lines.append(
                f"  FPR (finalized before EOS, truncated): {stats.fpr * 100:.1f}% "
                f"({num_early_final}/{num_non_error})"
            )
            lines.append(
                f"  No final captured (no TTFB, empty): "
                f"{stats.num_no_final / num_non_error * 100:.1f}% "
                f"({stats.num_no_final}/{num_non_error})"
            )
            lines.append(
                f"  Dropped from latency (total): "
                f"{stats.num_premature_eos / num_non_error * 100:.1f}% "
                f"({stats.num_premature_eos}/{num_non_error})"
            )

        if stats.ttfb_mean is not None:
            lines.append("  TTFB (seconds, over valid subset):")
            lines.append(f"    Mean:   {stats.ttfb_mean:.3f}")
            lines.append(f"    Median: {stats.ttfb_median:.3f}")
            lines.append(f"    Std:    {stats.ttfb_std:.3f}")
            lines.append(f"    Min:    {stats.ttfb_min:.3f}")
            lines.append(f"    Max:    {stats.ttfb_max:.3f}")
            lines.append(f"    P90:    {stats.ttfb_p90:.3f}")
            lines.append(f"    P95:    {stats.ttfb_p95:.3f}")
            lines.append(f"    P99:    {stats.ttfb_p99:.3f}")

            if stats.ttfb_by_duration:
                lines.append("  By Audio Duration:")
                for bucket, mean_ttfb in stats.ttfb_by_duration.items():
                    if mean_ttfb is not None:
                        lines.append(f"    {bucket}: {mean_ttfb:.3f}s")
        else:
            lines.append("  No successful benchmarks")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)
