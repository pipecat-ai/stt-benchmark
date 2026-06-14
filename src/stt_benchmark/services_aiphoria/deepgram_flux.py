"""Deepgram Flux via the /v2/listen WebSocket API.

Flux requires Deepgram's v2 endpoint with ``model=flux-general-en`` (not v1 /
``nova-3-*``). It self-endpoints via ``EndOfTurn`` TurnInfo events rather than
v1 ``Finalize`` or ``speech_final``. We suppress local VAD->Finalize and push a
single finalized ``TranscriptionFrame`` on the first non-empty ``EndOfTurn``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
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

try:
    from deepgram import AsyncDeepgramClient
    from deepgram.core.events import EventType
    from deepgram.listen.v2.types import ListenV2CloseStream, ListenV2TurnInfo
except ModuleNotFoundError as e:
    raise ImportError(
        "deepgram-sdk is required for deepgram_flux; install pipecat-ai[deepgram]"
    ) from e

FLUX_MODEL = "flux-general-en"
STREAM_CONNECT_TIMEOUT_SECS = 10.0


class DeepgramFluxSTTService(STTService):
    """Deepgram Flux cloud API (/v2/listen)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = FLUX_MODEL,
        encoding: str = "linear16",
        eager_eot_threshold: float | None = None,
        finalize_on_eager_eot: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            ttfs_p99_latency=2.0,
            stt_ttfb_timeout=8.0,
            settings=STTSettings(model=model, language=Language.EN),
            **kwargs,
        )
        self._api_key = api_key
        self._model = model
        self._encoding = encoding
        self._eager_eot_threshold = eager_eot_threshold
        self._finalize_on_eager_eot = finalize_on_eager_eot
        self._client = AsyncDeepgramClient(api_key=api_key)
        self._connection = None
        self._connection_task = None
        self._stream_ready = asyncio.Event()
        self._latest_transcript = ""
        self._final_sent = False

    @property
    def stream_ready_event(self) -> asyncio.Event:
        """Set once the Flux v2 websocket is connected and ready for audio."""
        return self._stream_ready

    def can_generate_metrics(self) -> bool:
        return True

    def _reset_utterance(self) -> None:
        self._latest_transcript = ""
        self._final_sent = False

    async def _push_final(self, text: str) -> None:
        if self._final_sent or not text.strip():
            return
        self._final_sent = True
        frame = TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language.EN)
        frame.finalized = True
        await self.push_frame(frame)

    def _parse_turn_info(self, message) -> ListenV2TurnInfo | None:
        if isinstance(message, ListenV2TurnInfo):
            return message
        if isinstance(message, dict) and message.get("type") == "TurnInfo":
            try:
                return ListenV2TurnInfo.model_validate(message)
            except Exception as e:
                logger.warning(f"{self}: failed to parse TurnInfo: {e}")
        return None

    async def _on_message(self, message) -> None:
        turn = self._parse_turn_info(message)
        if turn is None:
            return

        transcript = (turn.transcript or "").strip()
        if transcript:
            self._latest_transcript = transcript
            if turn.event in ("Update", "StartOfTurn", "EagerEndOfTurn", "TurnResumed"):
                await self.push_frame(
                    InterimTranscriptionFrame(
                        transcript, self._user_id, time_now_iso8601(), Language.EN
                    )
                )

        if (
            self._finalize_on_eager_eot
            and turn.event == "EagerEndOfTurn"
            and self._latest_transcript
        ):
            await self._push_final(self._latest_transcript)
        elif turn.event == "EndOfTurn" and self._latest_transcript:
            await self._push_final(self._latest_transcript)

    async def _on_error(self, error) -> None:
        logger.warning(f"{self} connection error, will retry: {error}")
        await self.push_error(error_msg=f"{error}")

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        self._stream_ready.clear()
        await self._connect()
        try:
            await asyncio.wait_for(
                self._stream_ready.wait(),
                timeout=STREAM_CONNECT_TIMEOUT_SECS,
            )
        except asyncio.TimeoutError as e:
            await self._disconnect()
            raise TimeoutError(
                f"{self}: Flux v2 websocket not ready within "
                f"{STREAM_CONNECT_TIMEOUT_SECS}s"
            ) from e
        logger.debug(f"{self}: Flux v2 websocket ready (sample_rate={self.sample_rate})")

    async def stop(self, frame: EndFrame) -> None:
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame) -> None:
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        if not self._connection:
            logger.trace(f"{self}: dropping audio chunk ({len(audio)} bytes), no connection")
            yield None
            return
        try:
            await self._connection.send_media(audio)
        except Exception as e:
            logger.warning(f"{self}: send_media failed, connection will reconnect: {e}")
            self._connection = None
        yield None

    async def _connect(self) -> None:
        logger.debug(f"{self}: Connecting to Deepgram Flux v2")
        self._connection_task = self.create_task(self._connection_handler())

    async def _disconnect(self) -> None:
        if not self._connection_task:
            return

        logger.debug(f"{self}: Disconnecting from Deepgram Flux v2")
        self._stream_ready.clear()
        connection = self._connection
        self._connection = None

        if connection:
            await connection.send_close_stream(ListenV2CloseStream(type="CloseStream"))

        await self.cancel_task(self._connection_task)
        self._connection_task = None

    async def _connection_handler(self) -> None:
        while True:
            try:
                connect_kwargs: dict[str, str | float] = {
                    "model": self._model,
                    "encoding": self._encoding,
                    "sample_rate": str(self.sample_rate),
                }
                if self._eager_eot_threshold is not None:
                    connect_kwargs["eager_eot_threshold"] = str(
                        self._eager_eot_threshold
                    )
                async with self._client.listen.v2.connect(**connect_kwargs) as connection:
                    self._connection = connection
                    connection.on(EventType.MESSAGE, self._on_message)
                    connection.on(EventType.ERROR, self._on_error)
                    self._stream_ready.set()
                    logger.debug(f"{self}: Flux v2 websocket initialized")
                    await connection.start_listening()
            except Exception as e:
                logger.warning(f"{self}: Connection lost, will retry: {e}")
            finally:
                self._stream_ready.clear()
                self._connection = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Flux self-endpoints; do not send v1-style Finalize on local VAD stop.
        await STTService.process_frame(self, frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._reset_utterance()
