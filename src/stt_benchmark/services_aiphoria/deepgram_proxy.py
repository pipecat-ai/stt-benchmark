"""Pipecat STT service for the production speech-proxy fronting Deepgram.

Setup 4: Deepgram nova-3 reached through our staging speech-proxy
(`speech-proxy.main.stage.aiphoria.pro:443`), which exposes the SAME
platform_proto v2 gRPC API as asr-backend-service. This is the real production
integration path, so it isolates the proxy/transport overhead on top of
Deepgram and uses Deepgram's UtteranceEnd EOU as the proxy delivers it.

Differences from AiphoriaSTTService (1a/1b), kept in this subclass so the shared
module stays byte-identical:
  * TLS secure_channel (host:443) instead of insecure localhost.
  * recognizer `asr_deepgram_en_nova3` (smoke-verified English nova-3;
    server-side config: interim_results + smart_format + punctuate +
    return_final_on_utterance_end, utterance_end_ms=1000).
  * The proxy signals end-of-turn via `is_final=True` only; `eou_reason` is
    always 0 (UNSPECIFIED). So we finalize on the FIRST is_final with non-empty
    text, eou_reason-agnostic (smoke showed exactly one final per utterance).

TTFS instrumentation is unchanged: Pipecat base STTService TTFB, started on the
shared local Silero VADUserStoppedSpeaking - 0.2s, stopped when we push the one
finalized TranscriptionFrame -> identical clock/anchor as every other setup.
"""

from __future__ import annotations

import grpc
from pipecat.frames.frames import InterimTranscriptionFrame, StartFrame
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from platform_proto.aiphoria.platform.asr_backend.v2.asr_pb2_grpc import ASRStub

from stt_benchmark.services_aiphoria.aiphoria_stt import AiphoriaSTTService

PROXY_TARGET = "speech-proxy.main.stage.aiphoria.pro:443"
PROXY_RECOGNIZER = "asr_deepgram_en_nova3"


class DeepgramProxySTTService(AiphoriaSTTService):
    """Deepgram via the production speech-proxy (gRPC v2, TLS)."""

    def __init__(self, *, recognizer: str = PROXY_RECOGNIZER, **kwargs) -> None:
        super().__init__(
            target=PROXY_TARGET,
            language_id=recognizer,
            mode="native_eou",  # run_stt only queues audio; we gate finals here
            **kwargs,
        )

    async def start(self, frame: StartFrame):
        # Mirror AiphoriaSTTService.start but with a TLS secure channel.
        # Call the grandparent (STTService) so the parent's insecure_channel
        # line is bypassed.
        await STTService.start(self, frame)
        import asyncio

        self._send_q = asyncio.Queue()
        self._channel = grpc.aio.secure_channel(
            self._target, grpc.ssl_channel_credentials()
        )
        self._stub = ASRStub(self._channel)
        self._reset_utterance()
        self._conn_task = self.create_task(
            self._connection_handler(), name="deepgram_proxy_grpc"
        )

    async def _on_reply(self, resp):
        text = (getattr(resp, "raw_text", "") or "").strip()
        is_final = bool(getattr(resp, "is_final", False))
        if is_final and text:
            await self._push_final(text)
        elif text:
            await self.push_frame(
                InterimTranscriptionFrame(
                    text, self._user_id, time_now_iso8601(), Language.EN
                )
            )
