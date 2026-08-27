"""Tests for multi-segment transcript aggregation.

Streaming STT services routinely emit several final ``TranscriptionFrame``s for
a single utterance (Pipecat's user aggregator concatenates these before handing
them to the LLM). The benchmark must aggregate them the same way, or WER is
computed against a truncated hypothesis.
"""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import patch

import pytest
from pipecat.frames.frames import Frame, TranscriptionFrame, VADUserStoppedSpeakingFrame
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601

from stt_benchmark.config import get_config
from stt_benchmark.models import AudioSample, ServiceName
from stt_benchmark.observers.transcription_collector import TranscriptionCollectorObserver
from stt_benchmark.pipeline.benchmark_runner import BenchmarkRunner

SEGMENTS = ["Hello there,", "how are you", "doing today?"]
EXPECTED = "Hello there, how are you doing today?"


def test_observer_concatenates_multiple_segments():
    """The collector joins successive final segments with a single space."""
    observer = TranscriptionCollectorObserver()
    observer.set_current_sample("sample-1")

    for segment in SEGMENTS:
        observer._handle_transcription(segment)

    assert observer.get_transcription_for_sample("sample-1") == EXPECTED


def test_observer_isolates_samples():
    """Segments from a previous sample never leak into the next one."""
    observer = TranscriptionCollectorObserver()

    observer.set_current_sample("sample-1")
    observer._handle_transcription("first utterance")

    observer.set_current_sample("sample-2")
    observer._handle_transcription("second utterance")

    assert observer.get_transcription_for_sample("sample-1") == "first utterance"
    assert observer.get_transcription_for_sample("sample-2") == "second utterance"


def test_observer_restarts_a_resumed_sample():
    """Re-running a sample discards the earlier attempt instead of appending."""
    observer = TranscriptionCollectorObserver()

    observer.set_current_sample("sample-1")
    observer._handle_transcription("stale attempt")

    observer.set_current_sample("sample-1")
    observer._handle_transcription("fresh attempt")

    assert observer.get_transcription_for_sample("sample-1") == "fresh attempt"


class MultiSegmentSTT(STTService):
    """Mock STT that emits several final TranscriptionFrames after VAD stop."""

    def __init__(self, segments: list[str], gap: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self._segments = segments
        self._gap = gap

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        yield None

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        await super()._handle_vad_user_stopped_speaking(frame)
        self.create_task(self._emit_segments())

    async def _emit_segments(self):
        await asyncio.sleep(0.3)  # mimic a realistic first-segment latency
        for i, segment in enumerate(self._segments):
            await self.push_frame(TranscriptionFrame(segment, "user", time_now_iso8601()))
            if i < len(self._segments) - 1:
                await asyncio.sleep(self._gap)


def _first_sample() -> AudioSample:
    """A real dataset sample, so Silero VAD sees actual speech."""
    import sqlite3

    db_path = get_config().results_db
    if not db_path.exists():
        pytest.skip("no benchmark database; run 'stt-benchmark download' first")

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT sample_id, audio_path, duration_seconds, dataset_index FROM samples "
            "WHERE duration_seconds BETWEEN 2 AND 5 LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    if not row or not Path(row[1]).exists():
        pytest.skip("no audio samples available locally")

    return AudioSample(
        sample_id=row[0], audio_path=row[1], duration_seconds=row[2], dataset_index=row[3]
    )


async def _run_with_mock(segments: list[str], gap: float):
    service = MultiSegmentSTT(segments=segments, gap=gap)
    with patch("stt_benchmark.pipeline.benchmark_runner.create_stt_service", return_value=service):
        return await BenchmarkRunner().benchmark_sample(_first_sample(), ServiceName.DEEPGRAM)


@pytest.mark.parametrize("gap", [0.1, 0.8])
async def test_pipeline_aggregates_multiple_segments(gap):
    """End-to-end: every segment inside the collection window reaches the result."""
    result = await _run_with_mock(SEGMENTS, gap)

    assert result.error is None
    assert result.transcription == EXPECTED


async def test_ttfs_measures_to_the_last_segment():
    """TTFS must reflect the final segment, not the first one."""
    gap = 0.4
    result = await _run_with_mock(SEGMENTS, gap)

    # Two gaps between three segments, so the last segment lands ~0.8s after
    # the first. TTFS should be well past the first segment's arrival.
    assert result.ttfb_seconds is not None
    assert result.ttfb_seconds > gap * (len(SEGMENTS) - 1)


async def test_slowly_trickling_segments_are_collected():
    """Segments keep the window open, up to the collection deadline.

    This used to be the known limit of the design: collection stopped a fixed
    2s after the *first* segment, so anything still in flight was lost. The
    window now follows the service and closes only once it has gone quiet, or
    once the deadline after the end of speech passes -- whichever comes first.
    """
    result = await _run_with_mock(SEGMENTS, gap=1.5)

    assert result.transcription == EXPECTED
