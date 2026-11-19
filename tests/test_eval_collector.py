"""When the eval bot decides a sample is done, and what it reports."""

import asyncio
import time

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import ErrorFrame, InputAudioRawFrame, InputDTMFFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService

from stt_benchmark.evals.collector import (
    SAMPLE_KEY,
    AudioEndObserver,
    SampleResult,
    STTErrorObserver,
    collect_sample,
)
from stt_benchmark.observers.metrics_collector import MetricsCollectorObserver
from stt_benchmark.observers.transcription_collector import TranscriptionCollectorObserver

# Short limits so the tests run in well under a second each.
FAST = {
    "quiet_secs": 0.1,
    "max_silence_secs": 0.3,
    "transcription_timeout_secs": 0.2,
    "ttfb_wait_secs": 0.05,
    "poll_secs": 0.005,
}


def _observers():
    audio_end = AudioEndObserver()
    transcripts = TranscriptionCollectorObserver()
    metrics = MetricsCollectorObserver()
    transcripts.set_current_sample(SAMPLE_KEY)
    metrics.set_current_sample(SAMPLE_KEY)
    return audio_end, transcripts, metrics


def _end_audio(audio_end: AudioEndObserver) -> None:
    audio_end.ended_at = time.monotonic()


def _ttfb(metrics: MetricsCollectorObserver, value: float) -> None:
    metrics.ttfb_by_sample[SAMPLE_KEY] = value
    metrics._ttfb_received.set()


def _pushed(frame) -> FramePushed:
    return FramePushed(
        source=IdentityFilter(),
        destination=IdentityFilter(),
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


async def test_the_marker_is_the_end_of_the_audio():
    audio_end = AudioEndObserver()

    await audio_end.on_push_frame(
        _pushed(InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1))
    )
    assert audio_end.ended_at is None

    marker = InputDTMFFrame(button=KeypadEntry.POUND)
    await audio_end.on_push_frame(_pushed(marker))
    first = audio_end.ended_at
    assert first is not None
    # The frame is pushed again by each processor it passes; the first time counts.
    await audio_end.on_push_frame(_pushed(marker))
    assert audio_end.ended_at == first


async def test_done_once_the_audio_ended_and_the_stt_is_quiet():
    audio_end, transcripts, metrics = _observers()
    _end_audio(audio_end)
    transcripts._handle_transcription("hello there")
    _ttfb(metrics, 0.31)

    started = time.monotonic()
    result = await collect_sample(audio_end, transcripts, metrics, **FAST)

    assert result == SampleResult(transcription="hello there", ttfb_seconds=0.31)
    assert 0.1 <= time.monotonic() - started < 0.3


async def test_a_long_pause_in_the_speech_does_not_end_the_sample():
    """A sound, then silence far longer than the quiet window, then the speech."""
    audio_end, transcripts, metrics = _observers()
    transcripts._handle_transcription("Hi.")

    async def speak_then_end():
        # Several quiet windows with no transcript: the audio is still playing.
        await asyncio.sleep(0.4)
        transcripts._handle_transcription("My subscription renews tomorrow.")
        _end_audio(audio_end)
        _ttfb(metrics, 0.27)

    rest = asyncio.create_task(speak_then_end())
    result = await collect_sample(audio_end, transcripts, metrics, **FAST)
    await rest

    assert result.transcription == "Hi. My subscription renews tomorrow."
    assert result.ttfb_seconds == 0.27
    assert not result.capped


async def test_each_transcript_reopens_the_quiet_window():
    audio_end, transcripts, metrics = _observers()
    _end_audio(audio_end)
    transcripts._handle_transcription("one")

    async def late():
        await asyncio.sleep(0.07)
        transcripts._handle_transcription("two")

    rest = asyncio.create_task(late())
    result = await collect_sample(audio_end, transcripts, metrics, **FAST)
    await rest

    assert result.transcription == "one two"


async def test_the_limit_ends_a_sample_still_delivering():
    audio_end, transcripts, metrics = _observers()
    _end_audio(audio_end)

    async def keep_talking():
        while True:
            transcripts._handle_transcription("more")
            await asyncio.sleep(0.03)

    chatter = asyncio.create_task(keep_talking())
    result = await collect_sample(audio_end, transcripts, metrics, **FAST)
    chatter.cancel()

    assert result.capped
    assert result.transcription.startswith("more more")


async def test_no_transcription_ends_after_both_limits():
    audio_end, transcripts, metrics = _observers()
    _end_audio(audio_end)

    started = time.monotonic()
    result = await collect_sample(audio_end, transcripts, metrics, **FAST)

    assert result == SampleResult(transcription=None, ttfb_seconds=None)
    assert time.monotonic() - started >= 0.5


async def test_a_late_ttfb_is_waited_for():
    audio_end, transcripts, metrics = _observers()
    _end_audio(audio_end)
    transcripts._handle_transcription("hello")

    async def late():
        await asyncio.sleep(0.13)
        _ttfb(metrics, 0.4)

    rest = asyncio.create_task(late())
    result = await collect_sample(
        audio_end, transcripts, metrics, **{**FAST, "ttfb_wait_secs": 0.5}
    )
    await rest

    assert result.ttfb_seconds == 0.4


def test_the_message_names_itself_and_carries_the_result():
    message = SampleResult(transcription="hi", ttfb_seconds=0.2, capped=True).as_message()
    assert message == {
        "type": "stt-benchmark-result",
        "transcription": "hi",
        "ttfb_seconds": 0.2,
        "capped": True,
        "error": None,
    }


class _SilentSTT(STTService):
    async def run_stt(self, audio):
        yield None


async def test_an_error_from_the_stt_service_is_noted_once():
    errors = STTErrorObserver()
    assert errors.first is None

    lost = ErrorFrame(error="connection error: timed out")
    # The service pushes it; the processors upstream then pass it along.
    await errors.on_push_frame(
        FramePushed(
            source=_SilentSTT(),
            destination=IdentityFilter(),
            frame=lost,
            direction=FrameDirection.UPSTREAM,
            timestamp=0,
        )
    )
    await errors.on_push_frame(_pushed(lost))

    assert errors.errors == ["connection error: timed out"]
    assert errors.first == "connection error: timed out"


async def test_a_sample_with_an_stt_error_reports_it_with_what_arrived():
    audio_end, transcripts, metrics = _observers()
    errors = STTErrorObserver()
    errors.errors.append("connection error: timed out")
    _end_audio(audio_end)
    transcripts._handle_transcription("hello")
    _ttfb(metrics, 1.2)

    result = await collect_sample(audio_end, transcripts, metrics, errors=errors, **FAST)

    assert result == SampleResult(
        transcription="hello", ttfb_seconds=1.2, error="connection error: timed out"
    )
    assert result.as_message()["error"] == "connection error: timed out"
