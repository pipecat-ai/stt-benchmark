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
- VAD state is sent as ``{"type": "vad_start"}`` and
  ``{"type": "vad_stop"}``.
- Interim results arrive as ``{"type": "transcript", "is_final": false, ...}``.
- Sending ``{"type": "reset", "finalize": true}`` triggers a hard reset:
  the server pads + finalizes, replies with a final (delta) transcript as
  ``{"type": "transcript", "is_final": true, "finalize": true, ...}`` and
  resets its state. One WebSocket connection is used per benchmark sample,
  so the final delta is the full utterance transcript.
"""

import asyncio
import contextlib
import json
import os
import re
import threading
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.utils.time import time_now_iso8601
from websockets.asyncio.client import connect as websocket_connect


_TELEMETRY_LOCK = threading.Lock()
_TELEMETRY_COUNTERS: dict[str, int] = {}
_DEFAULT_FINALIZE_SILENCE_MS = 2500
_MAX_FINALIZE_SILENCE_MS = 10_000


def _telemetry_run_tag() -> str | None:
    for env_name in (
        "NEMOTRON_RUN_TAG",
        "NEMOTRON_TELEMETRY_RUN_TAG",
        "STT_BENCHMARK_RUN_TAG",
    ):
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def _safe_tag_filename(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("._-") or "tag"


def _telemetry_dir() -> Path:
    configured = os.environ.get("NEMOTRON_TELEMETRY_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[2] / "stt_benchmark_data" / "client_telemetry"


def _next_telemetry_index(tag: str) -> int:
    with _TELEMETRY_LOCK:
        index = _TELEMETRY_COUNTERS.get(tag, 0)
        _TELEMETRY_COUNTERS[tag] = index + 1
        return index


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {value!r}") from e


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
        self._audio_send_lock = asyncio.Lock()
        self._finalize_mode = (
            os.environ.get("NEMOTRON_FINALIZE_MODE", "").strip().lower()
        )
        self._single_finalize_mode = self._finalize_mode == "single"
        self._finalize_silence_ms = _DEFAULT_FINALIZE_SILENCE_MS
        if self._single_finalize_mode:
            self._finalize_silence_ms = _env_int(
                "NEMOTRON_FINALIZE_SILENCE_MS", _DEFAULT_FINALIZE_SILENCE_MS
            )
        if self._single_finalize_mode and not (
            0 <= self._finalize_silence_ms < _MAX_FINALIZE_SILENCE_MS
        ):
            raise ValueError(
                "NEMOTRON_FINALIZE_SILENCE_MS must be >= 0 and < "
                f"{_MAX_FINALIZE_SILENCE_MS}"
            )
        self._finalize_silence_seconds = self._finalize_silence_ms / 1000
        self._single_finalize_lock = asyncio.Lock()
        self._single_finalize_timer_task: asyncio.Task | None = None
        self._single_finalize_stop_seq = 0
        self._single_finalize_pending_stop_seq: int | None = None
        self._single_finalize_reset_sent = False
        self._final_transcription_committed = False
        self._last_final_transcription_text: str | None = None
        self._telemetry_tag = _telemetry_run_tag()
        self._telemetry_batch_index = (
            _next_telemetry_index(self._telemetry_tag) if self._telemetry_tag else None
        )
        self._telemetry_written = False
        self._telemetry_hard_resets = 0
        self._telemetry_soft_resets = 0
        self._telemetry_final_frames = 0
        self._telemetry_vad_starts = 0
        self._telemetry_vad_stops = 0
        self._telemetry_early_final = False
        self._telemetry_started_after_final = 0
        self._telemetry_user_speaking = False

    def can_generate_metrics(self) -> bool:
        """Whether this service can generate processing metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the service and connect to the local ASR server."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Flush any buffered audio, then disconnect."""
        if not self._final_transcription_committed:
            if self._single_finalize_mode:
                await self._send_pending_single_finalize_on_stop()
            else:
                await self._send_finalize_reset()
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and disconnect."""
        await self._cancel_single_finalize_timer(clear_pending=True)
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
        await self._cancel_single_finalize_timer(clear_pending=True)
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()
        self._write_telemetry_once(reason="disconnect")

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
                async with self._audio_send_lock:
                    if self._websocket and self._ready:
                        await self._websocket.send(audio)
            except Exception as e:
                logger.error(f"{self} failed to send audio: {e}")
                await self._report_error(ErrorFrame(f"Failed to send audio: {e}"))
        yield None

    async def _send_reset(self, finalize: bool = True, before_send=None) -> bool:
        """Ask the server to finalize the current utterance (hard reset)."""
        if not (self._websocket and self._ready):
            return False
        try:
            async with self._audio_send_lock:
                if not (self._websocket and self._ready):
                    return False
                if before_send:
                    before_send()
                await self._websocket.send(
                    json.dumps({"type": "reset", "finalize": finalize})
                )
            if finalize:
                self._telemetry_hard_resets += 1
            else:
                self._telemetry_soft_resets += 1
            return True
        except Exception as e:
            logger.error(f"{self} failed to send reset: {e}")
            return False

    async def _send_vad_signal(self, signal_type: str) -> bool:
        """Send a declarative VAD state signal to the server."""
        if not (self._websocket and self._ready):
            return False
        try:
            async with self._audio_send_lock:
                if not (self._websocket and self._ready):
                    return False
                await self._websocket.send(json.dumps({"type": signal_type}))
            return True
        except Exception as e:
            logger.error(f"{self} failed to send {signal_type}: {e}")
            return False

    async def _send_finalize_reset(self) -> bool:
        """Send a hard reset, arming pipecat finalization only for that send."""
        sent = await self._send_reset(finalize=True, before_send=self.request_finalize)
        if not sent:
            self._finalize_requested = False
        return sent

    async def _arm_single_finalize_timer(self):
        """Arm/re-arm the single-finalize debounce timer for the latest VAD stop."""
        previous_task = None
        async with self._single_finalize_lock:
            if self._single_finalize_reset_sent:
                return
            previous_task = self._single_finalize_timer_task
            self._single_finalize_stop_seq += 1
            stop_seq = self._single_finalize_stop_seq
            self._single_finalize_pending_stop_seq = stop_seq
            self._single_finalize_timer_task = asyncio.create_task(
                self._single_finalize_timer(stop_seq),
                name=f"{self}::single_finalize_timer",
            )

        if previous_task and not previous_task.done():
            previous_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await previous_task

    async def _single_finalize_timer(self, stop_seq: int):
        try:
            await asyncio.sleep(self._finalize_silence_seconds)
            await self._send_armed_single_finalize(stop_seq)
        except asyncio.CancelledError:
            pass
        finally:
            async with self._single_finalize_lock:
                if self._single_finalize_timer_task is asyncio.current_task():
                    self._single_finalize_timer_task = None

    async def _cancel_single_finalize_timer(self, *, clear_pending: bool):
        """Cancel the debounce timer; optionally abandon the armed VAD stop."""
        async with self._single_finalize_lock:
            task = self._single_finalize_timer_task
            self._single_finalize_timer_task = None
            if clear_pending:
                self._single_finalize_pending_stop_seq = None
                self._single_finalize_stop_seq += 1

        if task and task is not asyncio.current_task() and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _send_pending_single_finalize_on_stop(self) -> bool:
        """Flush the currently armed single finalize during service shutdown."""
        async with self._single_finalize_lock:
            task = self._single_finalize_timer_task
            pending_stop_seq = self._single_finalize_pending_stop_seq
            self._single_finalize_timer_task = None

        if task and task is not asyncio.current_task() and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        if pending_stop_seq is None:
            return False
        return await self._send_armed_single_finalize(pending_stop_seq)

    async def _send_armed_single_finalize(self, stop_seq: int) -> bool:
        """Send the one hard reset for the VAD stop still armed by the timer."""
        async with self._single_finalize_lock:
            if self._single_finalize_reset_sent:
                return False
            if self._single_finalize_pending_stop_seq != stop_seq:
                return False
            self._single_finalize_pending_stop_seq = None
            self._single_finalize_reset_sent = True
            if self._single_finalize_timer_task is asyncio.current_task():
                self._single_finalize_timer_task = None

        if not (self._websocket and self._ready):
            async with self._single_finalize_lock:
                self._single_finalize_reset_sent = False
            return False

        sent = await self._send_finalize_reset()
        if not sent:
            async with self._single_finalize_lock:
                self._single_finalize_reset_sent = False
        return sent

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Trigger a hard reset when VAD detects end of speech.

        The benchmark pipeline emits ``VADUserStoppedSpeakingFrame`` (there is
        no turn aggregator, so no ``UserStoppedSpeakingFrame``). ``request_finalize``
        + ``confirm_finalize`` mark the next transcript as finalized so the base
        class reports TTFS at the moment it arrives.
        """
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._telemetry_vad_starts += 1
            self._telemetry_user_speaking = True
            if self._telemetry_final_frames:
                self._telemetry_early_final = True
                self._telemetry_started_after_final += 1
            if self._final_transcription_committed and not self._single_finalize_mode:
                self._final_transcription_committed = False
            if self._single_finalize_mode:
                await self._cancel_single_finalize_timer(clear_pending=True)

        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._send_vad_signal("vad_start")

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._telemetry_vad_stops += 1
            self._telemetry_user_speaking = False
            await self._send_vad_signal("vad_stop")
            if self._single_finalize_mode:
                await self._arm_single_finalize_timer()
            else:
                await self._send_finalize_reset()

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
        is_finalize_final = bool(data.get("is_final")) and data.get("finalize") is True
        if not text:
            if is_finalize_final:
                self._finalize_requested = False
            return
        if not is_finalize_final:
            await self.push_frame(
                InterimTranscriptionFrame(
                    text, self._user_id, time_now_iso8601(), None
                )
            )
            return

        if text == self._last_final_transcription_text:
            self._finalize_requested = False
            return
        if not self._finalize_requested:
            logger.debug(f"{self} ignoring unarmed finalize transcript: '{text}'")
            return

        # Hard-reset final: server already dedups to the new delta, which
        # (one connection per sample) is the full utterance transcript.
        self.confirm_finalize()
        self._last_final_transcription_text = text
        self._final_transcription_committed = True
        self._telemetry_final_frames += 1
        if self._telemetry_user_speaking:
            self._telemetry_early_final = True
        await self.push_frame(
            TranscriptionFrame(text, self._user_id, time_now_iso8601(), None)
        )

    def _write_telemetry_once(self, *, reason: str) -> Path | None:
        """Append one per-connection telemetry JSONL record when enabled."""
        if self._telemetry_written or not self._telemetry_tag:
            return None
        self._telemetry_written = True

        telemetry_dir = _telemetry_dir()
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        path = telemetry_dir / f"{_safe_tag_filename(self._telemetry_tag)}.jsonl"
        payload = {
            "run_tag": self._telemetry_tag,
            "benchmark_batch_index": self._telemetry_batch_index,
            "timestamp": time_now_iso8601(),
            "reason": reason,
            "hard_resets": self._telemetry_hard_resets,
            "soft_resets": self._telemetry_soft_resets,
            "final_transcription_frames": self._telemetry_final_frames,
            "early_final": self._telemetry_early_final,
            "vad_starts": self._telemetry_vad_starts,
            "vad_stops": self._telemetry_vad_stops,
            "started_after_final_count": self._telemetry_started_after_final,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
        return path
