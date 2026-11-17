"""Reading the eval suite's results back as benchmark results.

The suite appends one JSON line per finished sample to ``results.jsonl``. A
sample that ran to its end carries the bot's result message as what its one
expectation matched; one that did not carries the harness's failure.

A sample is a failure too when the STT service reported an error while it ran.
Its transcription and TTFS are then left out, so it counts as a sample without
a transcript and not towards WER or TTFS; what did arrive stays in the record.
"""

import json
from collections.abc import Iterator
from pathlib import Path

from stt_benchmark.evals.collector import RESULT_MESSAGE_TYPE
from stt_benchmark.models import AudioSample, BenchmarkResult, ServiceName


class ResultsReader:
    """Reads the records appended to a ``results.jsonl`` since the last read.

    The suite writes while a run is under way, so the reader is called
    repeatedly and picks up where it stopped. A line still being written is
    left for the next call.
    """

    def __init__(self, path: Path):
        """Initialize the reader.

        Args:
            path: The suite's ``results.jsonl``; it need not exist yet.
        """
        self._path = path
        self._offset = 0

    def read_new(self) -> Iterator[dict]:
        """The records appended since the last call, in order."""
        if not self._path.exists():
            return
        with self._path.open("rb") as f:
            f.seek(self._offset)
            for line in f:
                if not line.endswith(b"\n"):
                    break
                self._offset += len(line)
                if line.strip():
                    yield json.loads(line)


def benchmark_result(
    record: dict, samples: dict[str, AudioSample], model: str | None = None
) -> BenchmarkResult:
    """The benchmark result a suite record stands for.

    Args:
        record: One ``results.jsonl`` record of a benchmark run.
        samples: The run's samples by id, for each one's duration.
        model: The model label the run was started with, if any.

    Returns:
        The result, with the bot's transcription and TTFS when the sample ran
        to its end without the STT service reporting an error, or the failure
        as its error otherwise.
    """
    # The suite names a run "<scenario file>/<scenario>", here "<service>/<sample id>".
    sample_id = record["scenario"].split("/", 1)[-1]
    result = BenchmarkResult(
        sample_id=sample_id,
        service_name=ServiceName(record["name"]),
        model_name=model,
        audio_duration_seconds=samples[sample_id].duration_seconds,
    )

    message = _result_message(record) if record.get("passed") else None
    if message is None:
        result.error = _failure(record)
        return result
    if message.get("error"):
        result.error = f"stt error: {message['error']}"
        return result
    result.transcription = message.get("transcription")
    result.ttfb_seconds = message.get("ttfb_seconds")
    return result


def _result_message(record: dict) -> dict | None:
    """The bot's result message a passed record matched, or None when it holds none."""
    for turn in record.get("turns", []):
        for expectation in turn.get("expectations", []):
            try:
                message = json.loads(expectation.get("matched") or "")
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict) and message.get("type") == RESULT_MESSAGE_TYPE:
                return message
    return None


def _failure(record: dict) -> str:
    """Why a record holds no result, in the harness's words."""
    if record.get("error"):
        return str(record["error"])
    failures = record.get("failures") or []
    if failures:
        return "; ".join(f"{f.get('kind')}: {f.get('reason')}" for f in failures)
    if record.get("skipped"):
        return f"skipped: {record['skipped']}"
    return "the bot reported no result"
