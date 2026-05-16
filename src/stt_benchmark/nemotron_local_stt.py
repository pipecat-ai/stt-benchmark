"""Local Nemotron streaming STT service for the benchmark.

Connects over WebSocket to the self-hosted Nemotron streaming ASR server in
this repo (``src/nemotron_speech/server.py``), so the benchmark can measure
the *local* checkpoint instead of NVIDIA's hosted NIM.

This is a pipecat 1.2.x port of ``pipecat_bots/nvidia_stt.py`` (written for
pipecat 0.0.98). It is intentionally leaner than the original: the benchmark
pipeline has no turn aggregator and the 1.2.x ``STTService`` base measures
TTFS automatically (speech-end -> finalized ``TranscriptionFrame``), so the
old manual ``MetricsFrame`` emission and frame-ordering workarounds are gone.

Server protocol (see src/nemotron_speech/server.py):
- On connect the server sends ``{"type": "ready"}``.
- Audio is sent as raw 16-bit PCM, 16 kHz, mono.
- Interim results arrive as ``{"type": "transcript", "is_final": false, ...}``.
- Sending ``{"type": "reset", "finalize": true}`` triggers a hard reset:
  the server pads + finalizes, replies with a final (delta) transcript as
  ``{"type": "transcript", "is_final": true, "finalize": true, ...}`` and
  resets its state. One WebSocket connection is used per benchmark sample,
  so the final delta is the full utterance transcript.
"""

import json

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.utils.time import time_now_iso8601
from websockets.asyncio.client import connect as websocket_connect


class NemotronLocalSTTService(WebsocketSTTService):
    """Streaming STT against the local Nemotron WebSocket ASR server.

    A hard reset (``finalize=True``) is issued when VAD detects the user
    stopped speaking; the resulting final transcript is pushed as a finalized
    ``TranscriptionFrame`` so the base class reports TTFS immediately.
    """

    def __init__(
        self,
        *,
        url: str = "ws://localhost:8080",
        sample_rate: int = 16000,
        **kwargs,
    ):
        """Initialize the local Nemotron STT service.

        Args:
            url: WebSocket URL of the local Nemotron ASR server.
            sample_rate: Audio sample rate; the server requires 16000.
            **kwargs: Additional arguments passed to WebsocketSTTService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._url = url
        self._receive_task = None
        self._ready = False

    def can_generate_metrics(self) -> bool:
        """Whether this service can generate processing metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the service and connect to the local ASR server."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Flush any buffered audio, then disconnect."""
        await self._send_reset(finalize=True)
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and disconnect."""
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await self._connect_websocket()
        await super()._connect()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error)
            )

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Open the WebSocket and wait for the server's ready message."""
        try:
            if self._websocket:
                return
            logger.debug(f"{self} connecting to {self._url}")
            self._websocket = await websocket_connect(self._url)
            self._ready = False
            try:
                msg = await self._websocket.recv()
                if json.loads(msg).get("type") == "ready":
                    logger.debug(f"{self} server ready")
                self._ready = True
            except Exception as e:
                logger.warning(f"{self} no ready message ({e}); proceeding anyway")
                self._ready = True
            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self} connection failed: {e}")
            self._websocket = None
            raise

    async def _disconnect_websocket(self):
        self._ready = False
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.debug(f"{self} error closing websocket: {e}")
            finally:
                self._websocket = None
        await self._call_event_handler("on_disconnected")

    async def run_stt(self, audio: bytes):
        """Send audio to the server; results arrive via the receive task."""
        if self._websocket and self._ready:
            try:
                await self._websocket.send(audio)
            except Exception as e:
                logger.error(f"{self} failed to send audio: {e}")
                await self._report_error(ErrorFrame(f"Failed to send audio: {e}"))
        yield None

    async def _send_reset(self, finalize: bool = True):
        """Ask the server to finalize the current utterance (hard reset)."""
        if not (self._websocket and self._ready):
            return
        try:
            await self._websocket.send(
                json.dumps({"type": "reset", "finalize": finalize})
            )
        except Exception as e:
            logger.error(f"{self} failed to send reset: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Trigger a hard reset when VAD detects end of speech.

        The benchmark pipeline emits ``VADUserStoppedSpeakingFrame`` (there is
        no turn aggregator, so no ``UserStoppedSpeakingFrame``). ``request_finalize``
        + ``confirm_finalize`` mark the next transcript as finalized so the base
        class reports TTFS at the moment it arrives.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self.request_finalize()
            await self._send_reset(finalize=True)

    async def _receive_messages(self):
        if not self._websocket:
            return
        async for message in self._websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                logger.error(f"{self} invalid JSON: {e}")
                continue

            msg_type = data.get("type")
            if msg_type == "transcript":
                await self._handle_transcript(data)
            elif msg_type == "ready":
                self._ready = True
            elif msg_type == "error":
                err = data.get("message", "unknown error")
                logger.error(f"{self} server error: {err}")
                await self._report_error(ErrorFrame(f"Server error: {err}"))

    async def _handle_transcript(self, data: dict):
        text = (data.get("text") or "").strip()
        if not text:
            return
        if data.get("is_final"):
            # Hard-reset final: server already dedups to the new delta, which
            # (one connection per sample) is the full utterance transcript.
            self.confirm_finalize()
            await self.push_frame(
                TranscriptionFrame(text, self._user_id, time_now_iso8601(), None)
            )
        else:
            await self.push_frame(
                InterimTranscriptionFrame(
                    text, self._user_id, time_now_iso8601(), None
                )
            )
