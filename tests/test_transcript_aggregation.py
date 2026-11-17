"""Tests for multi-segment transcript aggregation.

Streaming STT services routinely emit several final ``TranscriptionFrame``s for
a single utterance (Pipecat's user aggregator concatenates these before handing
them to the LLM). The benchmark must aggregate them the same way, or WER is
computed against a truncated hypothesis.
"""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601
from pipecat.workers.runner import WorkerRunner

from stt_benchmark.config import get_config
from stt_benchmark.evals.bot import build_worker
from stt_benchmark.evals.collector import SampleResult
from stt_benchmark.observers.transcription_collector import TranscriptionCollectorObserver

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


class HarnessAudio(FrameProcessor):
    """Stands in for the eval harness: plays a sample in real time, then silence."""

    CHUNK_MS = 20

    def __init__(self, audio: bytes, sample_rate: int):
        super().__init__()
        self._audio = audio
        self._sample_rate = sample_rate
        self._playing: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._playing = self.create_task(self._play())

    async def _play(self):
        chunk = int(self._sample_rate * self.CHUNK_MS / 1000) * 2
        offset = 0
        while True:
            audio = self._audio[offset : offset + chunk].ljust(chunk, b"\x00")
            offset += chunk
            await self.push_frame(
                InputAudioRawFrame(audio=audio, sample_rate=self._sample_rate, num_channels=1)
            )
            await asyncio.sleep(self.CHUNK_MS / 1000)


class MessageSink(FrameProcessor):
    """Stands in for the transport's output: keeps the server messages the bot sends."""

    def __init__(self):
        super().__init__()
        self.messages: list = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, RTVIServerMessageFrame):
            self.messages.append(frame.data)
        await self.push_frame(frame, direction)


def _sample_audio() -> bytes:
    """A real dataset sample, so Silero VAD sees actual speech."""
    import sqlite3

    db_path = get_config().results_db
    if not db_path.exists():
        pytest.skip("no benchmark database; run 'stt-benchmark download' first")

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT audio_path FROM samples WHERE duration_seconds BETWEEN 2 AND 5 LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    if not row or not Path(row[0]).exists():
        pytest.skip("no audio samples available locally")

    return Path(row[0]).read_bytes()


async def _run_with_mock(segments: list[str], gap: float) -> tuple[SampleResult, list]:
    """Run the bot's pipeline on a sample, with a mock STT, as the harness would drive it."""
    config = get_config()
    sink = MessageSink()
    worker, report_sample = build_worker(
        HarnessAudio(_sample_audio(), config.sample_rate),
        MultiSegmentSTT(segments=segments, gap=gap),
        sink,
        vad_stop_secs=config.vad_stop_secs,
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        result = await asyncio.wait_for(report_sample(), timeout=40)
        # Let the message travel down the pipeline to the sink.
        await asyncio.sleep(0.2)
    finally:
        await runner.cancel()
        await asyncio.gather(running, return_exceptions=True)
    return result, sink.messages


@pytest.mark.parametrize("gap", [0.1, 0.8])
async def test_pipeline_aggregates_multiple_segments(gap):
    """End-to-end: every segment inside the collection window reaches the result."""
    result, messages = await _run_with_mock(SEGMENTS, gap)

    assert result.transcription == EXPECTED
    assert not result.capped
    # The harness gets the same result, as the bot's one server message.
    assert messages == [result.as_message()]


async def test_ttfs_measures_to_the_last_segment():
    """TTFS must reflect the final segment, not the first one."""
    gap = 0.4
    result, _ = await _run_with_mock(SEGMENTS, gap)

    # Two gaps between three segments, so the last segment lands ~0.8s after
    # the first. TTFS should be well past the first segment's arrival.
    assert result.ttfb_seconds is not None
    assert result.ttfb_seconds > gap * (len(SEGMENTS) - 1)


async def test_slowly_trickling_segments_are_collected():
    """Segments keep the window open, up to the collection deadline.

    Collection follows the service and closes only once it has gone quiet, or
    once the deadline after the end of speech passes -- whichever comes first.
    """
    result, _ = await _run_with_mock(SEGMENTS, gap=1.5)

    assert result.transcription == EXPECTED
