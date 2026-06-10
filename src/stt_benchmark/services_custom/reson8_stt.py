"""Reson8 speech-to-text service implementation.

Reson8 is not (yet) bundled with Pipecat, so this is a local Pipecat
``WebsocketSTTService`` implementation of its WSS "Turns" API:
https://docs.reson8.dev/api/speech-to-text/turns/

The Turns API is turn-based: the server performs its own endpoint/turn
detection and streams a small set of JSON control messages over the websocket
while we feed it raw binary PCM audio:

- ``{"type": "turn_start"}``           — speaker began a new turn
- ``{"type": "turn_end_candidate",     — detected silence; ``text`` is the
   "text": "..."}``                       (running) best transcript for the turn
- ``{"type": "turn_end"}``             — confirms the previous candidate is final
- ``{"type": "turn_continuation"}``    — cancels the candidate; speaker resumed,
                                          the next candidate carries the full
                                          accumulated text

We push each ``turn_end_candidate`` as an ``InterimTranscriptionFrame`` and emit
the final ``TranscriptionFrame`` on ``turn_end`` (using the latest candidate's
text, since ``turn_end`` itself carries none). TTFS is measured to the last
``turn_end_candidate`` — when the final text first became available.

Endpointing is driven entirely by the server; aggressiveness is tuned with the
``patience`` query parameters (see :func:`stt_benchmark.services.create_reson8`).
"""

import json
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from urllib.parse import urlencode

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import DEFAULT_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Reson8, you need to `pip install websockets`.")
    raise ImportError(f"Missing module: {e}") from e


DEFAULT_URL = "wss://api.reson8.dev/v1/speech-to-text/turns"


def language_to_reson8_language(language: Language) -> str | None:
    """Convert a Pipecat ``Language`` to a Reson8 language code.

    Reson8 takes an optional ISO language code as a transcription bias, so we
    map to the base two-letter code (e.g. ``Language.EN_US`` -> ``"en"``).
    """
    return resolve_language(language, {}, use_base_code=True)


@dataclass
class Reson8STTSettings(STTSettings):
    """Settings for :class:`Reson8STTService`."""


class Reson8STTService(WebsocketSTTService):
    """Speech-to-Text service using Reson8's WSS Turns API.

    Streams raw PCM audio over a websocket and turns Reson8's turn-based control
    messages into Pipecat ``InterimTranscriptionFrame`` / ``TranscriptionFrame``.

    For complete API documentation, see:
    https://docs.reson8.dev/api/speech-to-text/turns/
    """

    Settings = Reson8STTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = DEFAULT_URL,
        sample_rate: int | None = None,
        encoding: str = "pcm_s16le",
        num_channels: int = 1,
        custom_model_id: str | None = None,
        extra_headers: dict[str, str] | None = None,
        patience: float | None = None,
        min_patience_seconds: float | None = None,
        max_patience_seconds: float | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = DEFAULT_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Reson8 STT service.

        Args:
            api_key: Reson8 API key.
            url: Reson8 WSS Turns API URL.
            sample_rate: Audio sample rate in Hz. If None, taken from the pipeline.
            encoding: Audio encoding query param (``pcm_s16le`` or ``auto``).
            num_channels: Number of audio channels.
            custom_model_id: Optional custom model identifier.
            extra_headers: Additional websocket headers. Used to target a local
                piano build that authenticates via Tuba headers (``X-Customer-Id``
                etc.) injected by the gateway in production instead of the
                ``Authorization: ApiKey`` header.
            patience: End-of-turn patience scalar in [0, 1] (server query param).
            min_patience_seconds: Lower bound on the end-of-turn delay (query param).
            max_patience_seconds: Upper bound on the end-of-turn delay (query param).
            settings: Runtime-updatable settings (e.g. ``language``).
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
            **kwargs: Additional arguments passed to the STTService.
        """
        # reson8 has no selectable model (model is unset; a custom model is
        # selected via the custom_model_id query param), so model=None.
        default_settings = self.Settings(model=None, language=Language.EN)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._encoding = encoding
        self._num_channels = num_channels
        self._custom_model_id = custom_model_id
        self._extra_headers = extra_headers or {}
        self._patience = patience
        self._min_patience_seconds = min_patience_seconds
        self._max_patience_seconds = max_patience_seconds

        # Text of the most recent turn_end_candidate; emitted as the final
        # transcript on turn_end (which itself carries no text).
        self._candidate_text: str = ""

        # Wall-clock time of the most recent (non-empty) turn_end_candidate in the
        # current turn — when the final text first became available. TTFS is
        # measured to this rather than to turn_end.
        self._last_candidate_time: float = 0.0

        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Reson8 supports TTFB/TTFS metrics generation."""
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Pipecat ``Language`` to a Reson8 language code."""
        return language_to_reson8_language(language)

    async def start(self, frame: StartFrame):
        """Start the Reson8 websocket connection."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Reson8 websocket connection."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Reson8 websocket connection immediately."""
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio data to Reson8.

        Args:
            audio: Raw PCM audio bytes to transcribe.

        Yields:
            None — transcription results arrive asynchronously over the websocket.
        """
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(audio)
            except Exception as e:
                logger.warning(f"{self}: send failed: {e}")
        yield None

    def _build_url(self) -> str:
        """Build the websocket URL with Reson8 query parameters."""
        params = {
            "encoding": self._encoding,
            "sample_rate": self.sample_rate,
            "channels": self._num_channels,
        }
        language = self._settings.language
        if isinstance(language, Language):
            language = language_to_reson8_language(language)
        if language:
            params["language"] = language
        if self._custom_model_id:
            params["custom_model_id"] = self._custom_model_id
        if self._patience is not None:
            params["patience"] = self._patience
        if self._min_patience_seconds is not None:
            params["min_patience_seconds"] = self._min_patience_seconds
        if self._max_patience_seconds is not None:
            params["max_patience_seconds"] = self._max_patience_seconds
        return f"{self._url}?{urlencode(params)}"

    async def _connect(self):
        """Connect to Reson8 and start the receive task."""
        await self._connect_websocket()
        await super()._connect()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from Reson8 and tear down the receive task."""
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish the websocket connection to Reson8."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            url = self._build_url()
            logger.debug(f"Connecting to Reson8 STT at {url}")
            headers = {"Authorization": f"ApiKey {self._api_key}", **self._extra_headers}
            self._websocket = await websocket_connect(
                url,
                additional_headers=headers,
            )
            await self._call_event_handler("on_connected")
            logger.debug("Connected to Reson8 STT")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Reson8: {e}", exception=e)

    async def _disconnect_websocket(self):
        """Close the websocket connection to Reson8."""
        try:
            if self._websocket:
                logger.debug("Disconnecting from Reson8 STT")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Return the current websocket connection."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _emit_final_transcript(self):
        """Emit the latest candidate text as a final TranscriptionFrame.

        Latency is measured to the *last turn_end_candidate* time (when the final
        text first became available), not to turn_end. We report TTFB explicitly
        with that end time and cancel the pending timeout, then push the transcript
        as non-finalized so the base class does not re-report TTFB at push time
        (which would overwrite our earlier timestamp).
        """
        text = self._candidate_text.strip()
        if not text:
            return
        end_time = self._last_candidate_time if self._last_candidate_time > 0 else time.time()
        await self.stop_ttfb_metrics(end_time=end_time)
        await self._cancel_ttfb_timeout()
        await self.push_frame(
            TranscriptionFrame(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=self._settings.language
                if isinstance(self._settings.language, Language)
                else None,
                finalized=False,
            )
        )
        await self.stop_processing_metrics()
        self._candidate_text = ""
        self._last_candidate_time = 0.0

    async def _receive_messages(self):
        """Receive and process Reson8 turn control messages."""
        self._candidate_text = ""

        async for message in self._get_websocket():
            try:
                content = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"{self}: received non-JSON message: {message!r}")
                continue

            msg_type = content.get("type")

            if msg_type == "turn_start":
                await self.start_processing_metrics()
                self._candidate_text = ""
                self._last_candidate_time = 0.0

            elif msg_type == "turn_end_candidate":
                # Running best transcript for the turn — surfaced as interim, and
                # the latency endpoint (the time the final text became available).
                self._candidate_text = content.get("text", "") or ""
                if self._candidate_text:
                    self._last_candidate_time = time.time()
                    await self.push_frame(
                        InterimTranscriptionFrame(
                            text=self._candidate_text,
                            user_id=self._user_id,
                            timestamp=time_now_iso8601(),
                        )
                    )

            elif msg_type == "turn_continuation":
                # Speaker resumed; the prior candidate was premature. The next
                # turn_end_candidate carries the full accumulated text, so discard
                # the cancelled candidate's timestamp too.
                self._candidate_text = ""
                self._last_candidate_time = 0.0

            elif msg_type == "turn_end":
                # Turn confirmed — emit the latest candidate as the final segment.
                await self._emit_final_transcript()

            elif content.get("error") or content.get("code"):
                logger.error(f"{self}: Reson8 error: {content}")
