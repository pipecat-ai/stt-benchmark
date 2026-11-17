"""Deciding when a sample is done and what it produced.

The eval harness plays a sample to the bot and then keeps sending silence for
as long as it stays connected. Nothing tells the bot the audio has ended, and
a streaming STT does not say which transcript is its last, so the bot decides:
the sample is done once the VAD has heard the user stop and the STT has been
quiet for a while.
"""

import asyncio
import time
from dataclasses import asdict, dataclass

from loguru import logger
from pipecat.frames.frames import VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed

from stt_benchmark.observers.metrics_collector import MetricsCollectorObserver
from stt_benchmark.observers.transcription_collector import TranscriptionCollectorObserver

# The observers key what they collect by sample; a bot serves one sample.
SAMPLE_KEY = "sample"

# What the bot's server message calls itself, so a scenario can tell it from
# any other the bot might send.
RESULT_MESSAGE_TYPE = "stt-benchmark-result"


class VADStateObserver(BaseObserver):
    """Tracks whether the VAD hears the user speaking, and when it last heard them stop."""

    def __init__(self):
        super().__init__()
        self.speaking = False
        self.started = False
        # Monotonic time of the latest stop, None until the first one.
        self.stopped_at: float | None = None
        self._seen: set[int] = set()

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if not isinstance(frame, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            return
        # A frame is pushed once per processor it passes; count it once.
        if frame.id in self._seen:
            return
        self._seen.add(frame.id)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self.speaking = True
            self.started = True
        else:
            self.speaking = False
            self.stopped_at = time.monotonic()


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
    vad: VADStateObserver,
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

    The sample is done when the user has stopped speaking and ``quiet_secs``
    have passed since both that stop and the last transcript. ``max_silence_secs``
    after the stop the wait ends with what arrived, and when nothing has
    arrived by then, ``transcription_timeout_secs`` more are allowed for a
    first transcript. These are the limits the in-process runner applied from
    the end of the audio; here they run from the VAD's stop, ``stop_secs``
    later.

    A sample the VAD never hears still ends: once a transcript has arrived
    and ``max_silence_secs`` have passed with no other.

    Args:
        vad: The VAD's state.
        transcripts: The final transcripts collected so far.
        metrics: The TTFB metrics collected so far.
        quiet_secs: How long the STT must be quiet for the sample to be done.
        max_silence_secs: How long after the user stopped to keep collecting.
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
        now = time.monotonic()
        last_transcript = transcripts.get_last_transcription_time() or None

        if not vad.started:
            if last_transcript is not None and now - last_transcript >= max_silence_secs:
                logger.warning("The VAD heard no speech; ending on the STT's silence")
                break
            continue
        if vad.speaking or vad.stopped_at is None:
            continue

        since_stop = now - vad.stopped_at
        if last_transcript is None:
            if since_stop >= max_silence_secs + transcription_timeout_secs:
                logger.warning("No transcription arrived")
                break
            continue
        if since_stop >= quiet_secs and now - last_transcript >= quiet_secs:
            break
        if since_stop >= max_silence_secs:
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
