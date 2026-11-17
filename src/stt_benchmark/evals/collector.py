"""Deciding when a sample is done and what it produced.

The eval harness plays a sample to the bot and then keeps sending silence for
as long as it stays connected, and a streaming STT does not say which
transcript is its last. So the harness marks the end of the sample's audio with
a DTMF keypress, and the bot decides from there: the sample is done once the
audio has ended and the STT has been quiet for a while.

The keypress is only a marker. It travels as a data message, not as audio, and
nothing in the bot's pipeline acts on it but :class:`AudioEndObserver`, so the
STT service never hears of it.
"""

import asyncio
import time
from dataclasses import asdict, dataclass

from loguru import logger
from pipecat.frames.frames import InputDTMFFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed

from stt_benchmark.observers.metrics_collector import MetricsCollectorObserver
from stt_benchmark.observers.transcription_collector import TranscriptionCollectorObserver

# The observers key what they collect by sample; a bot serves one sample.
SAMPLE_KEY = "sample"

# What the bot's server message calls itself, so a scenario can tell it from
# any other the bot might send.
RESULT_MESSAGE_TYPE = "stt-benchmark-result"


class AudioEndObserver(BaseObserver):
    """Notes when the harness marked the end of the sample's audio.

    The harness sends a DTMF keypress once it has played the whole sample; see
    :func:`stt_benchmark.evals.manifest.write_suite`.
    """

    def __init__(self):
        super().__init__()
        # Monotonic time the marker arrived, None until it does.
        self.ended_at: float | None = None

    async def on_push_frame(self, data: FramePushed) -> None:
        if isinstance(data.frame, InputDTMFFrame) and self.ended_at is None:
            self.ended_at = time.monotonic()


@dataclass
class SampleResult:
    """What the bot measured for one sample, sent to the harness as its server message.

    Parameters:
        transcription: Every final transcript, joined with a space; None when
            the STT returned none.
        ttfb_seconds: The STT's TTFS as its service reported it; None when it
            reported none.
        capped: Whether the wait ended at its limit with the STT still
            delivering, so later transcripts may be missing.
    """

    transcription: str | None
    ttfb_seconds: float | None
    capped: bool = False

    def as_message(self) -> dict:
        """The result as the data of the bot's server message."""
        return {"type": RESULT_MESSAGE_TYPE, **asdict(self)}


async def collect_sample(
    audio_end: AudioEndObserver,
    transcripts: TranscriptionCollectorObserver,
    metrics: MetricsCollectorObserver,
    *,
    quiet_secs: float = 2.0,
    max_silence_secs: float = 10.0,
    transcription_timeout_secs: float = 10.0,
    ttfb_wait_secs: float = 2.5,
    poll_secs: float = 0.02,
) -> SampleResult:
    """Wait until the sample is done, then return what it produced.

    The sample is done when its audio has ended and ``quiet_secs`` have passed
    since both that end and the last transcript. ``max_silence_secs`` after the
    end the wait stops with what arrived, and when nothing has arrived by then,
    ``transcription_timeout_secs`` more are allowed for a first transcript.
    Nothing ends a sample before its audio has: a pause in the speech, however
    long, is part of the sample.

    Args:
        audio_end: When the sample's audio ended, once it has.
        transcripts: The final transcripts collected so far.
        metrics: The TTFB metrics collected so far.
        quiet_secs: How long the STT must be quiet for the sample to be done.
        max_silence_secs: How long after the audio ended to keep collecting.
        transcription_timeout_secs: How much longer to wait for a first transcript.
        ttfb_wait_secs: How long to wait for the TTFB metric once the sample
            is done; a service without finalized transcripts reports it on a
            timeout of its own.
        poll_secs: How often to look.

    Returns:
        The sample's result.
    """
    capped = False
    while True:
        await asyncio.sleep(poll_secs)
        if audio_end.ended_at is None:
            continue

        now = time.monotonic()
        since_end = now - audio_end.ended_at
        last_transcript = transcripts.get_last_transcription_time() or None
        if last_transcript is None:
            if since_end >= max_silence_secs + transcription_timeout_secs:
                logger.warning("No transcription arrived")
                break
            continue
        if since_end >= quiet_secs and now - last_transcript >= quiet_secs:
            break
        if since_end >= max_silence_secs:
            logger.warning("Transcripts were still arriving at the limit; later ones are lost")
            capped = True
            break

    if metrics.get_ttfb_for_sample(SAMPLE_KEY) is None:
        await metrics.wait_for_ttfb(timeout=ttfb_wait_secs)

    return SampleResult(
        transcription=transcripts.get_transcription_for_sample(SAMPLE_KEY),
        ttfb_seconds=metrics.get_ttfb_for_sample(SAMPLE_KEY),
        capped=capped,
    )
