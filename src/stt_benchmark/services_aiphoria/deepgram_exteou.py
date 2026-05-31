"""Deepgram ASR + our external VAD-EOU (benchmark setup 3).

Subclasses Pipecat's DeepgramSTTService but:
  * configures interim_results=True, endpointing=False, utterance_end_ms=None so
    Deepgram never decides finality itself;
  * treats EVERY Deepgram result (interim or is_final segment) as interim text,
    accumulating finalized segments + the live interim into a running transcript;
  * suppresses the stock VADUserStoppedSpeaking -> Deepgram Finalize behaviour;
  * gates the single final TranscriptionFrame on the SHARED ExternalVadEou
    (eou.py) - byte-identical to setup 1b, so 1b-vs-3 isolates ASR speed.

TTFS clock is unchanged: Pipecat base STTService starts TTFB on the local
Silero VADUserStoppedSpeaking - 0.2s and stops on our finalized frame.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from stt_benchmark.services_aiphoria.eou import ExternalVadEou

try:
    from deepgram.listen.v1.types import ListenV1Results
except Exception:  # pragma: no cover - import guard mirrors pipecat
    ListenV1Results = None


class DeepgramExternalEouSTTService(DeepgramSTTService):
    def __init__(self, *, api_key: str, max_silence_ms: int = 640, **kwargs):
        super().__init__(
            api_key=api_key,
            live_options=LiveOptions(
                model="nova-3-general",
                language=Language.EN,
                smart_format=False,
                profanity_filter=False,
                punctuate=False,
                interim_results=True,
                endpointing=False,
                utterance_end_ms=None,
            ),
            ttfs_p99_latency=1.0,
            stt_ttfb_timeout=8.0,
            **kwargs,
        )
        self._max_silence_ms = max_silence_ms
        self._dg_finals: list[str] = []
        self._dg_interim = ""
        self._final_sent = False
        self._eou: ExternalVadEou | None = None

    @property
    def _latest_text(self) -> str:
        parts = list(self._dg_finals)
        if self._dg_interim:
            parts.append(self._dg_interim)
        return " ".join(p for p in parts if p).strip()

    def _reset_utterance(self):
        self._dg_finals = []
        self._dg_interim = ""
        self._final_sent = False
        self._eou = ExternalVadEou(max_silence_ms=self._max_silence_ms)

    async def _push_final(self, text: str):
        if self._final_sent or not text.strip():
            return
        self._final_sent = True
        frame = TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language.EN)
        frame.finalized = True
        await self.push_frame(frame)

    async def _on_message(self, message):
        if ListenV1Results is None or not isinstance(message, ListenV1Results):
            return
        if not message.channel or len(message.channel.alternatives) == 0:
            return
        transcript = (message.channel.alternatives[0].transcript or "").strip()
        if not transcript:
            return
        if message.is_final:
            self._dg_finals.append(transcript)
            self._dg_interim = ""
        else:
            self._dg_interim = transcript
        await self.push_frame(
            InterimTranscriptionFrame(
                self._latest_text, self._user_id, time_now_iso8601(), Language.EN
            )
        )

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        # Send media to Deepgram via the inherited implementation.
        async for _ in DeepgramSTTService.run_stt(self, audio):
            pass
        if self._eou is None:
            self._reset_utterance()
        if self._eou.feed(audio) and self._latest_text:
            await self._push_final(self._latest_text)
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Skip DeepgramSTTService.process_frame (it sends a Finalize on VAD stop).
        # Call the grandparent so TTFB-start + audio routing still work.
        await STTService.process_frame(self, frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._reset_utterance()
