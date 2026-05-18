#!/usr/bin/env python3
"""Measurement scaffolding for Nemotron local STT experiments.

This script intentionally lives outside the benchmark framework. It reads the
locked benchmark DB, writes sidecar JSON files, and provides the fixed slices
and paired statistics needed by the Nemotron finalization work.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sqlite3
import tarfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "stt_benchmark_data"
DB_PATH = DATA_DIR / "results.db"
SLICES_DIR = DATA_DIR / "slices"
RUN_METADATA_DIR = DATA_DIR / "run_metadata"
CLIENT_TELEMETRY_DIR = DATA_DIR / "client_telemetry"
DEFAULT_SERVICE = "nemotron_local"
BASELINE_TAG = ""
SLICE_A_PATH = SLICES_DIR / "slice_A_dataset_index_0_199.json"
SLICE_B_PATH = SLICES_DIR / "slice_B_duration_stratified_seed1234.json"
VAD_PREFLIGHT_PATH = DATA_DIR / "vad_preflight_silero_stop0.2.json"
DEFAULT_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    dataset_index: int
    duration_seconds: float
    audio_path: str


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def display_tag(tag: str) -> str:
    return "''" if tag == "" else tag


def normalize_tag(tag: str) -> str:
    if tag in {"<empty>", "__empty__", "empty-string"}:
        return ""
    return tag


def safe_tag_filename(tag: str) -> str:
    if tag == "":
        return "empty-string"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("._-") or "tag"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def connect_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_samples(conn: sqlite3.Connection) -> list[Sample]:
    rows = conn.execute(
        """
        SELECT sample_id, dataset_index, duration_seconds, audio_path
        FROM samples
        ORDER BY dataset_index
        """
    ).fetchall()
    return [
        Sample(
            sample_id=row["sample_id"],
            dataset_index=int(row["dataset_index"]),
            duration_seconds=float(row["duration_seconds"]),
            audio_path=row["audio_path"],
        )
        for row in rows
    ]


def sample_payload(name: str, samples: list[Sample], extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "generated_at": now_iso(),
        "count": len(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "samples": [
            {
                "sample_id": sample.sample_id,
                "dataset_index": sample.dataset_index,
                "duration_seconds": sample.duration_seconds,
            }
            for sample in samples
        ],
        **extra,
    }


def duration_stratified_slice(
    samples: list[Sample],
    *,
    size: int = 200,
    bins: int = 10,
    seed: int = 1234,
) -> list[Sample]:
    if size > len(samples):
        raise ValueError(f"cannot draw {size} samples from {len(samples)}")
    if bins <= 0:
        raise ValueError("bins must be positive")

    ordered = sorted(samples, key=lambda s: (s.duration_seconds, s.dataset_index, s.sample_id))
    base_bin_size, extra_bin_count = divmod(len(ordered), bins)
    bin_sizes = [base_bin_size + (1 if i < extra_bin_count else 0) for i in range(bins)]

    base_quota, extra_quota_count = divmod(size, bins)
    quotas = [base_quota + (1 if i < extra_quota_count else 0) for i in range(bins)]

    rng = random.Random(seed)
    selected: list[Sample] = []
    remaining: list[Sample] = []
    offset = 0
    for bin_size, quota in zip(bin_sizes, quotas, strict=True):
        bin_samples = ordered[offset : offset + bin_size]
        offset += bin_size
        shuffled = list(bin_samples)
        rng.shuffle(shuffled)
        take = min(quota, len(shuffled))
        selected.extend(shuffled[:take])
        remaining.extend(shuffled[take:])

    if len(selected) < size:
        rng.shuffle(remaining)
        selected.extend(remaining[: size - len(selected)])

    return sorted(selected, key=lambda s: s.dataset_index)


def ensure_slices(conn: sqlite3.Connection, *, force: bool = False) -> dict[str, Path]:
    samples = load_samples(conn)

    if force or not SLICE_A_PATH.exists():
        slice_a = [sample for sample in samples if 0 <= sample.dataset_index <= 199]
        write_json(
            SLICE_A_PATH,
            sample_payload(
                "slice-A",
                slice_a,
                {
                    "method": "canonical dataset_index 0-199",
                    "dataset_index_start": 0,
                    "dataset_index_end": 199,
                },
            ),
        )

    if force or not SLICE_B_PATH.exists():
        slice_b = duration_stratified_slice(samples, size=200, bins=10, seed=1234)
        write_json(
            SLICE_B_PATH,
            sample_payload(
                "slice-B",
                slice_b,
                {
                    "method": (
                        "seeded duration-stratified draw: 10 equal-count duration bins, "
                        "20 sampled per bin"
                    ),
                    "seed": 1234,
                    "duration_bins": 10,
                },
            ),
        )

    return {"slice-A": SLICE_A_PATH, "slice-B": SLICE_B_PATH}


def load_slice(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    return [str(sample_id) for sample_id in payload["sample_ids"]]


def load_metrics_for_tag(
    conn: sqlite3.Connection,
    *,
    service: str,
    tag: str,
) -> dict[str, float]:
    rows = conn.execute(
        """
        SELECT wm.sample_id, wm.wer
        FROM wer_metrics wm
        JOIN ground_truth gt ON gt.sample_id = wm.sample_id
        WHERE wm.service_name = ? AND wm.model_name = ?
        """,
        (service, tag),
    ).fetchall()
    return {str(row["sample_id"]): float(row["wer"]) for row in rows}


def bootstrap_ci(values: list[float], *, seed: int, iterations: int) -> tuple[float, float] | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0], values[0]
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    sample_indices = rng.integers(0, len(arr), size=(iterations, len(arr)))
    means = arr[sample_indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def format_ci(ci: tuple[float, float] | None, *, suffix: str = "") -> str:
    if ci is None:
        return "[n/a]"
    return f"[{ci[0]:.4f}{suffix}, {ci[1]:.4f}{suffix}]"


def score_slice(
    *,
    slice_name: str,
    sample_ids: list[str],
    tag: str,
    tag_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    values = [tag_metrics[sample_id] * 100.0 for sample_id in sample_ids if sample_id in tag_metrics]
    paired_values = [
        (tag_metrics[sample_id] - baseline_metrics[sample_id]) * 100.0
        for sample_id in sample_ids
        if sample_id in tag_metrics and sample_id in baseline_metrics
    ]
    return {
        "slice": slice_name,
        "tag": tag,
        "slice_count": len(sample_ids),
        "tag_count": len(values),
        "paired_count": len(paired_values),
        "mean_wer_percent": float(np.mean(values)) if values else None,
        "mean_wer_ci95_percent": bootstrap_ci(values, seed=seed, iterations=iterations),
        "paired_delta_pp_vs_baseline": float(np.mean(paired_values)) if paired_values else None,
        "paired_delta_ci95_pp_vs_baseline": bootstrap_ci(
            paired_values, seed=seed + 17, iterations=iterations
        ),
    }


def print_score_report(
    conn: sqlite3.Connection,
    *,
    service: str,
    tags: list[str],
    baseline_tag: str,
    iterations: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    ensure_slices(conn)
    slices = {
        "slice-A": load_slice(SLICE_A_PATH),
        "slice-B": load_slice(SLICE_B_PATH),
    }
    baseline_metrics = load_metrics_for_tag(conn, service=service, tag=baseline_tag)

    print(f"service={service} baseline={display_tag(baseline_tag)} bootstrap={iterations} seed={seed}")
    results: dict[str, list[dict[str, Any]]] = {}
    for tag_index, tag in enumerate(tags):
        tag_metrics = load_metrics_for_tag(conn, service=service, tag=tag)
        print(f"\ntag={display_tag(tag)} rows={len(tag_metrics)}")
        tag_results: list[dict[str, Any]] = []
        for slice_index, (slice_name, sample_ids) in enumerate(slices.items()):
            result = score_slice(
                slice_name=slice_name,
                sample_ids=sample_ids,
                tag=tag,
                tag_metrics=tag_metrics,
                baseline_metrics=baseline_metrics,
                seed=seed + tag_index * 101 + slice_index * 1009,
                iterations=iterations,
            )
            tag_results.append(result)
            mean = result["mean_wer_percent"]
            delta = result["paired_delta_pp_vs_baseline"]
            mean_text = "n/a" if mean is None else f"{mean:.4f}%"
            delta_text = "n/a" if delta is None else f"{delta:+.4f} pp"
            print(
                "  "
                f"{slice_name}: n={result['tag_count']}/{result['slice_count']} "
                f"mean={mean_text} ci95={format_ci(result['mean_wer_ci95_percent'], suffix='%')} "
                f"paired_n={result['paired_count']} "
                f"delta_vs_baseline={delta_text} "
                f"ci95={format_ci(result['paired_delta_ci95_pp_vs_baseline'], suffix=' pp')}"
            )
        results[tag] = tag_results
    return results


def metadata_path_for_tag(tag: str) -> Path:
    return RUN_METADATA_DIR / f"{safe_tag_filename(tag)}.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_config_from_nemo(nemo_path: Path) -> tuple[str | None, str | None]:
    if not nemo_path.exists() or not nemo_path.is_file():
        return None, None
    try:
        with tarfile.open(nemo_path) as tar:
            for member in tar.getmembers():
                if Path(member.name).name in {"model_config.yaml", "model_config.yml"}:
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    data = extracted.read()
                    return sha256_bytes(data), member.name
    except tarfile.TarError:
        return None, None
    return None, None


def get_hf_revision(model_id: str, requested_revision: str | None) -> tuple[str | None, str | None]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - environment dependent
        return requested_revision, f"huggingface_hub unavailable: {exc}"

    try:
        info = HfApi().model_info(model_id, revision=requested_revision)
    except Exception as exc:  # pragma: no cover - network/cache dependent
        return requested_revision, str(exc)
    return getattr(info, "sha", None) or requested_revision, None


def find_cached_nemo_path(model_id: str, revision: str | None) -> str | None:
    try:
        from huggingface_hub import scan_cache_dir
    except Exception:
        return None

    try:
        cache_info = scan_cache_dir()
    except Exception:
        return None

    for repo in cache_info.repos:
        if repo.repo_id != model_id:
            continue
        for rev in repo.revisions:
            if revision and rev.commit_hash != revision:
                continue
            for file_info in rev.files:
                path = Path(file_info.file_path)
                if path.suffix == ".nemo" and path.exists():
                    return str(path.resolve())
    return None


def record_metadata(
    *,
    tag: str,
    model: str,
    hf_revision: str | None,
    nemo_path: str | None,
    config_path: str | None,
    config_json: str | None,
    right_context: int,
    decoding: str,
) -> Path:
    model_path = Path(model)
    is_local_model = model.endswith(".nemo") or model_path.exists()

    resolved_hf_revision: str | None = None
    hf_error: str | None = None
    resolved_nemo_path: str | None = None
    if is_local_model:
        resolved_nemo_path = str(model_path.resolve())
    else:
        resolved_hf_revision, hf_error = get_hf_revision(model, hf_revision)
        resolved_nemo_path = nemo_path or find_cached_nemo_path(model, resolved_hf_revision)

    if nemo_path:
        resolved_nemo_path = str(Path(nemo_path).expanduser().resolve())

    config_hash: str
    config_hash_source: str
    config_payload: dict[str, Any] | None = None
    if config_json:
        config_hash = sha256_bytes(config_json.encode("utf-8"))
        config_hash_source = "explicit-config-json"
        try:
            config_payload = json.loads(config_json)
        except json.JSONDecodeError:
            config_payload = {"raw": config_json}
    elif config_path:
        data = Path(config_path).expanduser().read_bytes()
        config_hash = sha256_bytes(data)
        config_hash_source = str(Path(config_path).expanduser().resolve())
    elif resolved_nemo_path:
        config_hash_from_nemo, source = hash_config_from_nemo(Path(resolved_nemo_path))
        if config_hash_from_nemo:
            config_hash = config_hash_from_nemo
            config_hash_source = f"{resolved_nemo_path}:{source}"
        else:
            runtime_payload = {
                "model": model,
                "hf_revision": resolved_hf_revision,
                "resolved_nemo_path": resolved_nemo_path,
                "right_context": right_context,
                "decoding": decoding,
            }
            config_hash = sha256_bytes(json.dumps(runtime_payload, sort_keys=True).encode("utf-8"))
            config_hash_source = "runtime-config-fallback"
            config_payload = runtime_payload
    else:
        runtime_payload = {
            "model": model,
            "hf_revision": resolved_hf_revision,
            "right_context": right_context,
            "decoding": decoding,
        }
        config_hash = sha256_bytes(json.dumps(runtime_payload, sort_keys=True).encode("utf-8"))
        config_hash_source = "runtime-config-fallback"
        config_payload = runtime_payload

    payload = {
        "tag": tag,
        "generated_at": now_iso(),
        "model": model,
        "hf_revision": resolved_hf_revision,
        "hf_revision_error": hf_error,
        "resolved_nemo_path": resolved_nemo_path,
        "right_context": right_context,
        "decoding": decoding,
        "model_config_hash": config_hash,
        "model_config_hash_source": config_hash_source,
        "model_config_payload": config_payload,
        "env": {
            "NEMOTRON_DECODING": os.environ.get("NEMOTRON_DECODING"),
            "NEMOTRON_ONSET_WARMUP_MS": os.environ.get("NEMOTRON_ONSET_WARMUP_MS"),
            "NEMOTRON_LOCAL_URL": os.environ.get("NEMOTRON_LOCAL_URL"),
        },
    }

    path = metadata_path_for_tag(tag)
    write_json(path, payload)
    return path


def load_ttfb_values(
    conn: sqlite3.Connection,
    *,
    service: str,
    tag: str,
) -> list[float]:
    rows = conn.execute(
        """
        SELECT ttfb_seconds
        FROM results
        WHERE service_name = ? AND model_name = ? AND ttfb_seconds IS NOT NULL
        ORDER BY sample_id
        """,
        (service, tag),
    ).fetchall()
    return [float(row["ttfb_seconds"]) for row in rows]


def telemetry_path_for_tag(tag: str) -> Path:
    return CLIENT_TELEMETRY_DIR / f"{safe_tag_filename(tag)}.jsonl"


def load_telemetry(tag: str, telemetry_dir: Path = CLIENT_TELEMETRY_DIR) -> list[dict[str, Any]]:
    path = telemetry_dir / f"{safe_tag_filename(tag)}.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def print_ttfb_and_counters(
    conn: sqlite3.Connection,
    *,
    service: str,
    tag: str,
    telemetry_tag: str | None,
) -> dict[str, Any]:
    ttfb = load_ttfb_values(conn, service=service, tag=tag)
    telemetry = load_telemetry(telemetry_tag if telemetry_tag is not None else tag)
    hard_resets = [float(row.get("hard_resets", 0)) for row in telemetry]
    final_frames = [float(row.get("final_transcription_frames", 0)) for row in telemetry]
    early_final_count = sum(1 for row in telemetry if row.get("early_final"))

    payload = {
        "service": service,
        "tag": tag,
        "ttfb_count": len(ttfb),
        "ttfb_median_ms": median(ttfb) * 1000.0 if ttfb else None,
        "ttfb_p95_ms": percentile(ttfb, 95) * 1000.0 if ttfb else None,
        "telemetry_count": len(telemetry),
        "hard_resets_per_sample": float(np.mean(hard_resets)) if hard_resets else None,
        "final_transcription_frames_per_sample": float(np.mean(final_frames))
        if final_frames
        else None,
        "early_final_count": early_final_count,
    }

    ttfb_median = payload["ttfb_median_ms"]
    ttfb_p95 = payload["ttfb_p95_ms"]
    print(f"tag={display_tag(tag)} observer_TTFS_n={len(ttfb)}")
    if ttfb:
        print(f"  TTFS median={ttfb_median:.1f} ms p95={ttfb_p95:.1f} ms")
    else:
        print("  TTFS median=n/a p95=n/a")
    print(
        "  "
        f"telemetry_n={len(telemetry)} "
        f"hard_resets/sample={payload['hard_resets_per_sample']} "
        f"final_frames/sample={payload['final_transcription_frames_per_sample']} "
        f"early_final_count={early_final_count}"
    )
    return payload


def resolve_audio_path(audio_path: str) -> Path:
    path = Path(audio_path)
    if path.is_absolute():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    return Path.cwd() / path


def summarize_gaps(gaps: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(gaps),
        "min_seconds": min(gaps) if gaps else None,
        "p50_seconds": percentile(gaps, 50),
        "p90_seconds": percentile(gaps, 90),
        "p95_seconds": percentile(gaps, 95),
        "p99_seconds": percentile(gaps, 99),
        "max_seconds": max(gaps) if gaps else None,
    }


def quiet_pipecat_logs() -> None:
    try:
        from loguru import logger as loguru_logger
    except Exception:
        return
    loguru_logger.disable("pipecat")


async def detect_vad_events_for_sample(
    sample: Sample,
    *,
    analyzer: Any,
    vad_state_type: Any,
    params: Any,
    sample_rate: int,
    chunk_ms: int,
    flush_silence_seconds: float,
) -> dict[str, Any]:
    analyzer._model.reset_states()
    analyzer._vad_buffer = b""
    analyzer._prev_volume = 0
    analyzer.set_params(params)

    audio_path = resolve_audio_path(sample.audio_path)
    audio = audio_path.read_bytes()
    chunk_bytes = int(sample_rate * chunk_ms / 1000) * 2
    flush_silence = b"\x00\x00" * int(sample_rate * flush_silence_seconds)
    stream = audio + flush_silence
    controller_state = vad_state_type.QUIET
    events: list[dict[str, Any]] = []

    for offset in range(0, len(stream), chunk_bytes):
        chunk = stream[offset : offset + chunk_bytes]
        if len(chunk) < chunk_bytes:
            chunk = chunk + (b"\x00" * (chunk_bytes - len(chunk)))
        new_state = await analyzer.analyze_audio(chunk)
        if (
            new_state != controller_state
            and new_state != vad_state_type.STARTING
            and new_state != vad_state_type.STOPPING
        ):
            timestamp = min((offset + len(chunk)) / 2 / sample_rate, sample.duration_seconds)
            if new_state == vad_state_type.SPEAKING:
                events.append({"type": "start", "timestamp_seconds": round(timestamp, 6)})
            elif new_state == vad_state_type.QUIET:
                events.append({"type": "stop", "timestamp_seconds": round(timestamp, 6)})
            controller_state = new_state

    segments: list[dict[str, float | None]] = []
    pending_start: float | None = None
    gaps: list[float] = []
    last_stop: float | None = None
    for event in events:
        timestamp = float(event["timestamp_seconds"])
        if event["type"] == "start":
            if last_stop is not None:
                gaps.append(round(max(0.0, timestamp - last_stop), 6))
                last_stop = None
            pending_start = timestamp
        elif event["type"] == "stop":
            if pending_start is not None:
                segments.append({"start_seconds": pending_start, "stop_seconds": timestamp})
                pending_start = None
            last_stop = timestamp

    if pending_start is not None:
        segments.append({"start_seconds": pending_start, "stop_seconds": None})

    return {
        "sample_id": sample.sample_id,
        "dataset_index": sample.dataset_index,
        "duration_seconds": sample.duration_seconds,
        "events": events,
        "speech_segments": segments,
        "start_count": sum(1 for event in events if event["type"] == "start"),
        "stop_count": sum(1 for event in events if event["type"] == "stop"),
        "stop_to_next_start_gaps_seconds": gaps,
        "max_stop_to_next_start_gap_seconds": max(gaps) if gaps else None,
    }


async def build_vad_preflight(
    conn: sqlite3.Connection,
    *,
    force: bool = False,
    sample_rate: int = 16000,
    chunk_ms: int = 20,
    stop_secs: float = 0.2,
) -> dict[str, Any]:
    if VAD_PREFLIGHT_PATH.exists() and not force:
        return json.loads(VAD_PREFLIGHT_PATH.read_text())

    quiet_pipecat_logs()
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams, VADState

    params = VADParams(stop_secs=stop_secs)
    analyzer = SileroVADAnalyzer(sample_rate=sample_rate, params=params)
    analyzer.set_sample_rate(sample_rate)
    flush_silence_seconds = max(1.0, stop_secs + 0.5)
    samples = load_samples(conn)

    per_sample: list[dict[str, Any]] = []
    all_gaps: list[float] = []
    for index, sample in enumerate(samples, start=1):
        sample_payload = await detect_vad_events_for_sample(
            sample,
            analyzer=analyzer,
            vad_state_type=VADState,
            params=params,
            sample_rate=sample_rate,
            chunk_ms=chunk_ms,
            flush_silence_seconds=flush_silence_seconds,
        )
        per_sample.append(sample_payload)
        all_gaps.extend(sample_payload["stop_to_next_start_gaps_seconds"])
        if index % 100 == 0:
            print(f"  VAD preflight processed {index}/{len(samples)} samples")

    multi_start_samples = sum(1 for sample in per_sample if sample["start_count"] > 1)
    payload = {
        "generated_at": now_iso(),
        "source_db": str(DB_PATH),
        "vad": {
            "analyzer": "SileroVADAnalyzer",
            "params": params.model_dump(),
            "sample_rate": sample_rate,
            "chunk_ms": chunk_ms,
            "flush_silence_seconds": flush_silence_seconds,
        },
        "summary": {
            "sample_count": len(per_sample),
            "multi_start_samples": multi_start_samples,
            "gap_distribution": summarize_gaps(all_gaps),
        },
        "samples": per_sample,
    }
    write_json(VAD_PREFLIGHT_PATH, payload)
    return payload


def print_vad_summary(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    gap_dist = summary["gap_distribution"]
    print(
        "VAD preflight: "
        f"samples={summary['sample_count']} "
        f"multi_start={summary['multi_start_samples']} "
        f"gaps={gap_dist['count']} "
        f"max_gap={gap_dist['max_seconds']} s "
        f"p95_gap={gap_dist['p95_seconds']} s"
    )
    print(f"  sidecar={VAD_PREFLIGHT_PATH}")


def telemetry_smoke(tag: str) -> Path:
    os.environ["NEMOTRON_RUN_TAG"] = tag
    quiet_pipecat_logs()
    from stt_benchmark.nemotron_local_stt import NemotronLocalSTTService

    service = NemotronLocalSTTService(url="ws://127.0.0.1:9")
    service._telemetry_vad_starts = 2
    service._telemetry_vad_stops = 2
    service._telemetry_hard_resets = 1
    service._telemetry_final_frames = 1
    service._telemetry_early_final = True
    service._telemetry_started_after_final = 1
    path = service._write_telemetry_once(reason="measure.py-smoke")
    if path is None:
        raise RuntimeError("telemetry smoke did not write a JSONL line")
    return path


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Path to results.db")
    parser.add_argument("--service", default=DEFAULT_SERVICE, help="Service name")


def cmd_slices(args: argparse.Namespace) -> None:
    with connect_db(args.db) as conn:
        paths = ensure_slices(conn, force=args.force)
    for name, path in paths.items():
        payload = json.loads(path.read_text())
        print(f"{name}: {payload['count']} ids -> {path}")


def cmd_score(args: argparse.Namespace) -> None:
    tags = [normalize_tag(tag) for tag in args.tags]
    with connect_db(args.db) as conn:
        print_score_report(
            conn,
            service=args.service,
            tags=tags,
            baseline_tag=normalize_tag(args.baseline_tag),
            iterations=args.bootstrap,
            seed=args.seed,
        )


def cmd_metadata(args: argparse.Namespace) -> None:
    path = record_metadata(
        tag=normalize_tag(args.tag),
        model=args.model,
        hf_revision=args.hf_revision,
        nemo_path=args.nemo_path,
        config_path=args.config_path,
        config_json=args.config_json,
        right_context=args.right_context,
        decoding=args.decoding,
    )
    print(f"metadata: tag={display_tag(normalize_tag(args.tag))} -> {path}")


def cmd_ttfb(args: argparse.Namespace) -> None:
    with connect_db(args.db) as conn:
        print_ttfb_and_counters(
            conn,
            service=args.service,
            tag=normalize_tag(args.tag),
            telemetry_tag=args.telemetry_tag,
        )


def cmd_vad_preflight(args: argparse.Namespace) -> None:
    with connect_db(args.db) as conn:
        payload = asyncio.run(
            build_vad_preflight(
                conn,
                force=args.force,
                sample_rate=args.sample_rate,
                chunk_ms=args.chunk_ms,
                stop_secs=args.stop_secs,
            )
        )
    print_vad_summary(payload)


def cmd_telemetry_smoke(args: argparse.Namespace) -> None:
    path = telemetry_smoke(args.tag)
    print(f"telemetry smoke: tag={args.tag} -> {path}")


def cmd_self_check(args: argparse.Namespace) -> None:
    with connect_db(args.db) as conn:
        print("Persisting canonical slices")
        paths = ensure_slices(conn, force=args.force_slices)
        for name, path in paths.items():
            payload = json.loads(path.read_text())
            print(f"  {name}: {payload['count']} ids -> {path}")

        print("\nScoring DB rows")
        print_score_report(
            conn,
            service=args.service,
            tags=[normalize_tag(tag) for tag in args.tags],
            baseline_tag=BASELINE_TAG,
            iterations=args.bootstrap,
            seed=args.seed,
        )

        print("\nObserver TTFS + counters")
        print_ttfb_and_counters(conn, service=args.service, tag=BASELINE_TAG, telemetry_tag=None)

        print("\nVAD preflight")
        vad_payload = asyncio.run(
            build_vad_preflight(
                conn,
                force=args.force_vad,
                sample_rate=args.sample_rate,
                chunk_ms=args.chunk_ms,
                stop_secs=args.stop_secs,
            )
        )
        print_vad_summary(vad_payload)

    print("\nRun metadata")
    metadata_path = record_metadata(
        tag=args.metadata_tag,
        model=args.model,
        hf_revision=args.hf_revision,
        nemo_path=args.nemo_path,
        config_path=args.config_path,
        config_json=args.config_json,
        right_context=args.right_context,
        decoding=args.decoding,
    )
    print(f"  metadata -> {metadata_path}")

    print("\nTelemetry smoke")
    smoke_path = telemetry_smoke(args.telemetry_smoke_tag)
    print(f"  telemetry -> {smoke_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    slices = subparsers.add_parser("slices", help="Persist fixed slice sidecars")
    add_common_args(slices)
    slices.add_argument("--force", action="store_true", help="Rewrite existing slice sidecars")
    slices.set_defaults(func=cmd_slices)

    score = subparsers.add_parser("score", help="Score one or more model_name tags")
    add_common_args(score)
    score.add_argument("--tags", nargs="+", default=["", "rc1ref"], help="model_name tags")
    score.add_argument("--baseline-tag", default="", help="Live baseline model_name tag")
    score.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap iterations")
    score.add_argument("--seed", type=int, default=1234, help="Bootstrap seed")
    score.set_defaults(func=cmd_score)

    metadata = subparsers.add_parser("metadata", help="Write run metadata sidecar")
    metadata.add_argument("--tag", required=True, help="Run/model tag")
    metadata.add_argument("--model", default=os.environ.get("NEMOTRON_MODEL", DEFAULT_MODEL))
    metadata.add_argument("--hf-revision", default=os.environ.get("NEMOTRON_HF_REVISION"))
    metadata.add_argument("--nemo-path", default=os.environ.get("NEMOTRON_RESOLVED_NEMO_PATH"))
    metadata.add_argument("--config-path", default=os.environ.get("NEMOTRON_MODEL_CONFIG_PATH"))
    metadata.add_argument("--config-json", default=os.environ.get("NEMOTRON_MODEL_CONFIG_JSON"))
    metadata.add_argument("--right-context", type=int, default=int(os.environ.get("NEMOTRON_RIGHT_CONTEXT", "1")))
    metadata.add_argument("--decoding", default=os.environ.get("NEMOTRON_DECODING", "greedy"))
    metadata.set_defaults(func=cmd_metadata)

    ttfb = subparsers.add_parser("ttfs", help="Read observer TTFS and telemetry counters")
    add_common_args(ttfb)
    ttfb.add_argument("--tag", default="", help="DB model_name tag")
    ttfb.add_argument("--telemetry-tag", default=None, help="Telemetry JSONL tag")
    ttfb.set_defaults(func=cmd_ttfb)

    vad = subparsers.add_parser("vad-preflight", help="Run offline Silero VAD preflight")
    add_common_args(vad)
    vad.add_argument("--force", action="store_true", help="Rewrite existing VAD sidecar")
    vad.add_argument("--sample-rate", type=int, default=16000)
    vad.add_argument("--chunk-ms", type=int, default=20)
    vad.add_argument("--stop-secs", type=float, default=0.2)
    vad.set_defaults(func=cmd_vad_preflight)

    smoke = subparsers.add_parser("telemetry-smoke", help="Emit one client telemetry JSONL line")
    smoke.add_argument("--tag", default="measure_smoke")
    smoke.set_defaults(func=cmd_telemetry_smoke)

    self_check = subparsers.add_parser("self-check", help="Run the Step-1 measurement self-check")
    add_common_args(self_check)
    self_check.add_argument("--tags", nargs="+", default=["", "rc1ref"], help="Tags to score")
    self_check.add_argument("--bootstrap", type=int, default=5000)
    self_check.add_argument("--seed", type=int, default=1234)
    self_check.add_argument("--force-slices", action="store_true")
    self_check.add_argument("--force-vad", action="store_true")
    self_check.add_argument("--sample-rate", type=int, default=16000)
    self_check.add_argument("--chunk-ms", type=int, default=20)
    self_check.add_argument("--stop-secs", type=float, default=0.2)
    self_check.add_argument("--metadata-tag", default="measure_selfcheck")
    self_check.add_argument("--telemetry-smoke-tag", default="measure_smoke")
    self_check.add_argument("--model", default=os.environ.get("NEMOTRON_MODEL", DEFAULT_MODEL))
    self_check.add_argument("--hf-revision", default=os.environ.get("NEMOTRON_HF_REVISION"))
    self_check.add_argument("--nemo-path", default=os.environ.get("NEMOTRON_RESOLVED_NEMO_PATH"))
    self_check.add_argument("--config-path", default=os.environ.get("NEMOTRON_MODEL_CONFIG_PATH"))
    self_check.add_argument("--config-json", default=os.environ.get("NEMOTRON_MODEL_CONFIG_JSON"))
    self_check.add_argument(
        "--right-context", type=int, default=int(os.environ.get("NEMOTRON_RIGHT_CONTEXT", "1"))
    )
    self_check.add_argument("--decoding", default=os.environ.get("NEMOTRON_DECODING", "greedy"))
    self_check.set_defaults(func=cmd_self_check)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        args = parser.parse_args(["self-check"])
    args.func(args)


if __name__ == "__main__":
    main()
