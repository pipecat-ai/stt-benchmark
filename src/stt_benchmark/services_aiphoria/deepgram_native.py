"""Setup 2b: Deepgram ASR + Deepgram's OWN native endpointing.

Unlike stock pipecat Setup 2 (where the harness's local Silero VAD fires a
forced `Finalize` at speech_end+0.2s), here Deepgram itself decides the
end-of-utterance: `endpointing=<ms>` makes Deepgram emit a result with
`speech_final=True` after that much trailing silence. We:
  * suppress pipecat's VAD-stop -> Finalize (process_frame override);
  * accumulate Deepgram is_final segments + live interim as the transcript;
  * take the FIRST `speech_final=True` (non-empty) as THE endpoint and push the
    single finalized TranscriptionFrame there.

TTFS clock is unchanged (Pipecat base STTService: start = local Silero
VADUserStoppedSpeaking - 0.2s; stop = our finalized frame), so the number is
directly comparable to Setups 1a/1b/2/3. It answers "if Deepgram decides the
turn ended on its own, how long after speech end is the final?" — note this is
config-dependent (scales with `endpointing_ms`).
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

try:
    from deepgram.listen.v1.types import ListenV1Results
except Exception:  # pragma: no cover
    ListenV1Results = None


class DeepgramNativeEouSTTService(DeepgramSTTService):
    def __init__(self, *, api_key: str, endpointing_ms: int = 300, **kwargs):
        super().__init__(
            api_key=api_key,
            live_options=LiveOptions(
                model="nova-3-general",
                language=Language.EN,
                smart_format=False,
                profanity_filter=False,
                punctuate=False,
                interim_results=True,
                endpointing=endpointing_ms,
                utterance_end_ms=None,
            ),
            ttfs_p99_latency=1.0,
            stt_ttfb_timeout=8.0,
            **kwargs,
        )
        self._endpointing_ms = endpointing_ms
        self._dg_finals: list[str] = []
        self._dg_interim = ""
        self._final_sent = False

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
        speech_final = bool(getattr(message, "speech_final", False))
        if transcript:
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
        # Deepgram's OWN end-of-utterance signal.
        if speech_final and self._latest_text:
            await self._push_final(self._latest_text)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Skip DeepgramSTTService.process_frame so NO Finalize is sent on the
        # local VAD stop; grandparent keeps TTFB-start + audio routing.
        await STTService.process_frame(self, frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._reset_utterance()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        async for _ in DeepgramSTTService.run_stt(self, audio):
            pass
        yield None
