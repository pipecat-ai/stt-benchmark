"""Running a benchmark through the eval suite and storing what it produces."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from pipecat.evals.suite import EvalManifest, EvalSuite

from stt_benchmark.evals.manifest import write_suite
from stt_benchmark.evals.results import ResultsReader, benchmark_result
from stt_benchmark.models import AudioSample, BenchmarkResult, ServiceName

if TYPE_CHECKING:
    from stt_benchmark.storage.database import Database

# How often the suite's results file is read for samples that have finished.
STORE_INTERVAL_SECS = 0.5


async def run_suite(
    run_dir: Path,
    samples_by_service: dict[ServiceName, list[AudioSample]],
    db: "Database",
    *,
    concurrency: int,
    vad_stop_secs: float | None = None,
    model: str | None = None,
    on_result: Callable[[BenchmarkResult], None] | None = None,
) -> list[BenchmarkResult]:
    """Benchmark each service's samples with the eval suite, storing every result as it arrives.

    Args:
        run_dir: Where the run's manifest, scenarios, logs and ``results.jsonl`` go.
        samples_by_service: The samples each service is to transcribe.
        db: The database; a result is stored as soon as its sample finishes,
            so an interrupted run keeps what it had.
        concurrency: How many samples run at once across all services.
        vad_stop_secs: The VAD's stop silence, or None for the configured default.
        model: The model label to store with each result, if any.
        on_result: Called with each result once it is stored.

    Returns:
        The run's results, in the order their samples finished.
    """
    manifest_path = write_suite(
        run_dir, samples_by_service, concurrency=concurrency, vad_stop_secs=vad_stop_secs
    )
    suite = EvalSuite(EvalManifest.load(manifest_path))

    samples = {s.sample_id: s for group in samples_by_service.values() for s in group}
    reader = ResultsReader(run_dir / "results.jsonl")
    results: list[BenchmarkResult] = []

    async def store_new() -> None:
        for record in reader.read_new():
            result = benchmark_result(record, samples, model)
            await db.insert_result(result)
            results.append(result)
            if on_result is not None:
                on_result(result)

    finished = asyncio.Event()

    async def store_as_they_finish() -> None:
        while not finished.is_set():
            await store_new()
            try:
                await asyncio.wait_for(finished.wait(), STORE_INTERVAL_SECS)
            except TimeoutError:
                pass

    storing = asyncio.create_task(store_as_they_finish())
    try:
        await suite.run(run_dir / "logs", results_path=run_dir / "results.jsonl")
    finally:
        finished.set()
        await storing
        await store_new()
    return results
