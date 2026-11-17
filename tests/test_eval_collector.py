"""When the eval bot decides a sample is done, and what it reports."""

import asyncio
import time

from stt_benchmark.evals.collector import (
    SAMPLE_KEY,
    SampleResult,
    VADStateObserver,
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
    vad = VADStateObserver()
    transcripts = TranscriptionCollectorObserver()
    metrics = MetricsCollectorObserver()
    transcripts.set_current_sample(SAMPLE_KEY)
    metrics.set_current_sample(SAMPLE_KEY)
    return vad, transcripts, metrics


def _start(vad: VADStateObserver) -> None:
    vad.speaking = True
    vad.started = True


def _stop(vad: VADStateObserver) -> None:
    vad.speaking = False
    vad.stopped_at = time.monotonic()


def _ttfb(metrics: MetricsCollectorObserver, value: float) -> None:
    metrics.ttfb_by_sample[SAMPLE_KEY] = value
    metrics._ttfb_received.set()


async def test_done_once_the_user_stopped_and_the_stt_is_quiet():
    vad, transcripts, metrics = _observers()
    _start(vad)
    _stop(vad)
    transcripts._handle_transcription("hello there")
    _ttfb(metrics, 0.31)

    started = time.monotonic()
    result = await collect_sample(vad, transcripts, metrics, **FAST)

    assert result == SampleResult(transcription="hello there", ttfb_seconds=0.31)
    assert 0.1 <= time.monotonic() - started < 0.3


async def test_a_transcript_during_a_pause_does_not_end_the_sample():
    vad, transcripts, metrics = _observers()
    _start(vad)
    _stop(vad)
    transcripts._handle_transcription("I'd like to book a table um")

    async def resume_and_finish():
        await asyncio.sleep(0.05)
        _start(vad)
        # Longer than the quiet window: speech with no transcript in it.
        await asyncio.sleep(0.25)
        _stop(vad)
        transcripts._handle_transcription("for four people on Friday")
        _ttfb(metrics, 0.45)

    rest = asyncio.create_task(resume_and_finish())
    result = await collect_sample(vad, transcripts, metrics, **FAST)
    await rest

    assert result.transcription == "I'd like to book a table um for four people on Friday"
    assert result.ttfb_seconds == 0.45
    assert not result.capped


async def test_each_transcript_reopens_the_quiet_window():
    vad, transcripts, metrics = _observers()
    _start(vad)
    _stop(vad)
    transcripts._handle_transcription("one")

    async def late():
        await asyncio.sleep(0.07)
        transcripts._handle_transcription("two")

    rest = asyncio.create_task(late())
    result = await collect_sample(vad, transcripts, metrics, **FAST)
    await rest

    assert result.transcription == "one two"


async def test_the_limit_ends_a_sample_still_delivering():
    vad, transcripts, metrics = _observers()
    _start(vad)
    _stop(vad)

    async def keep_talking():
        while True:
            transcripts._handle_transcription("more")
            await asyncio.sleep(0.03)

    chatter = asyncio.create_task(keep_talking())
    result = await collect_sample(vad, transcripts, metrics, **FAST)
    chatter.cancel()

    assert result.capped
    assert result.transcription.startswith("more more")


async def test_no_transcription_ends_after_both_limits():
    vad, transcripts, metrics = _observers()
    _start(vad)
    _stop(vad)

    started = time.monotonic()
    result = await collect_sample(vad, transcripts, metrics, **FAST)

    assert result == SampleResult(transcription=None, ttfb_seconds=None)
    assert time.monotonic() - started >= 0.5


async def test_a_sample_the_vad_never_hears_ends_on_the_stts_silence():
    vad, transcripts, metrics = _observers()
    transcripts._handle_transcription("quiet words")

    result = await collect_sample(vad, transcripts, metrics, **FAST)

    assert result == SampleResult(transcription="quiet words", ttfb_seconds=None)


async def test_a_late_ttfb_is_waited_for():
    vad, transcripts, metrics = _observers()
    _start(vad)
    _stop(vad)
    transcripts._handle_transcription("hello")

    async def late():
        await asyncio.sleep(0.13)
        _ttfb(metrics, 0.4)

    rest = asyncio.create_task(late())
    result = await collect_sample(vad, transcripts, metrics, **{**FAST, "ttfb_wait_secs": 0.5})
    await rest

    assert result.ttfb_seconds == 0.4


def test_the_message_names_itself_and_carries_the_result():
    message = SampleResult(transcription="hi", ttfb_seconds=0.2, capped=True).as_message()
    assert message == {
        "type": "stt-benchmark-result",
        "transcription": "hi",
        "ttfb_seconds": 0.2,
        "capped": True,
    }
