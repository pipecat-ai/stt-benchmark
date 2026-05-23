"""Pipecat STT service bridging to the deployed asr-backend-service (gRPC v2).

Used by benchmark setups:
  * 1a  mode="native_eou"   -> final TranscriptionFrame when the backend emits
        is_final with eou_reason in {organic(2), repeating_hypothesis(3)} and
        non-empty text. This is our true product EOU (in-Triton VAD-EOU,
        use_vad_eou=true, max_silence_ms=640).
  * 1b  mode="external_eou" -> every backend hypothesis is interim; the single
        final is gated by the shared ExternalVadEou (eou.py), identical to
        setup 3, so 1b-vs-3 isolates ASR speed only.

TTFS is produced by Pipecat's base STTService instrumentation: TTFB starts on
the pipeline's local Silero VADUserStoppedSpeaking - stop_secs (0.2s) and stops
when we push a finalized TranscriptionFrame. We only push ONE final per
utterance and mark it finalized so the base reports immediately.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import grpc
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from platform_proto.aiphoria.platform.asr_backend.v2.asr_pb2 import (
    AsrRequest,
    AsrRequestConfig,
    AudioMetadata,
)
from platform_proto.aiphoria.platform.asr_backend.v2.asr_pb2_grpc import ASRStub
from platform_proto.aiphoria.platform.constants.v2.audio_pb2 import Audio
from platform_proto.aiphoria.platform.constants.v2.streaming_pb2 import EndOfStream

from stt_benchmark.services_aiphoria.eou import ExternalVadEou

EOU_ORGANIC = 2
EOU_REPEATING = 3
PRODUCT_EOU_REASONS = (EOU_ORGANIC, EOU_REPEATING)


class AiphoriaSTTService(STTService):
    """Streams audio to asr-backend-service AudioToText (gRPC v2)."""

    def __init__(
        self,
        *,
        target: str = "localhost:50052",
        language_id: str = "en",
        audio_format: str = "PCM_S16_LE_16000",
        mode: str = "native_eou",
        max_silence_ms: int = 640,
        **kwargs,
    ) -> None:
        super().__init__(
            stt_ttfb_timeout=8.0,
            ttfs_p99_latency=1.0,
            settings=STTSettings(model="aiphoria-ctc-160ms", language=Language.EN),
            **kwargs,
        )
        if mode not in ("native_eou", "external_eou"):
            raise ValueError(mode)
        self._target = target
        self._language_id = language_id
        self._audio_format = audio_format
        self._mode = mode
        self._max_silence_ms = max_silence_ms
        self._channel: grpc.aio.Channel | None = None
        self._send_q: asyncio.Queue | None = None
        self._conn_task = None
        self._latest_text = ""
        self._final_sent = False
        self._eou: ExternalVadEou | None = None

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._send_q = asyncio.Queue()
        self._channel = grpc.aio.insecure_channel(self._target)
        self._stub = ASRStub(self._channel)
        self._reset_utterance()
        self._conn_task = self.create_task(self._connection_handler(), name="aiphoria_grpc")

    async def stop(self, frame):
        await super().stop(frame)
        await self._shutdown()

    async def cancel(self, frame):
        await super().cancel(frame)
        await self._shutdown()

    async def _shutdown(self):
        if self._send_q is not None:
            self._send_q.put_nowait(None)  # sentinel -> EndOfStream
        if self._conn_task is not None:
            await self.cancel_task(self._conn_task)
            self._conn_task = None
        if self._channel is not None:
            await self._channel.close()
            self._channel = None

    def _reset_utterance(self):
        self._latest_text = ""
        self._final_sent = False
        self._eou = ExternalVadEou(max_silence_ms=self._max_silence_ms)

    async def _req_iter(self) -> AsyncGenerator[AsrRequest, None]:
        yield AsrRequest(
            request_config=AsrRequestConfig(
                audio_metadata=AudioMetadata(format=self._audio_format),
                language_id=self._language_id,
            )
        )
        while True:
            item = await self._send_q.get()
            if item is None:
                yield AsrRequest(end_of_stream=EndOfStream())
                return
            yield AsrRequest(audio_data=Audio(data=item))

    async def _connection_handler(self):
        try:
            async for resp in self._stub.AudioToText(self._req_iter()):
                await self._on_reply(resp)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{self}: gRPC stream ended: {e}")

    async def _push_final(self, text: str):
        if self._final_sent or not text.strip():
            return
        self._final_sent = True
        frame = TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language.EN)
        frame.finalized = True
        await self.push_frame(frame)

    async def _on_reply(self, resp):
        text = (getattr(resp, "raw_text", "") or "").strip()
        is_final = bool(getattr(resp, "is_final", False))
        reason = int(getattr(resp, "eou_reason", 0) or 0)

        if self._mode == "native_eou":
            if is_final and reason in PRODUCT_EOU_REASONS and text:
                await self._push_final(text)
            elif text and not is_final:
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text, self._user_id, time_now_iso8601(), Language.EN
                    )
                )
        else:  # external_eou: all hypotheses interim; eou.py gates the final
            if text:
                self._latest_text = text
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text, self._user_id, time_now_iso8601(), Language.EN
                    )
                )

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        if self._send_q is not None:
            self._send_q.put_nowait(audio)
        if self._mode == "external_eou" and self._eou is not None:
            if self._eou.feed(audio) and self._latest_text:
                await self._push_final(self._latest_text)
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._reset_utterance()
