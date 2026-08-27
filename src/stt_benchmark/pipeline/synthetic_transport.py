"""Synthetic input transport for STT benchmarking.

Provides a BaseInputTransport implementation that reads audio from
bytes/files and pumps frames into a Pipecat Pipeline with real-time pacing.
"""

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import InputAudioRawFrame, StartFrame
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import TransportParams


class SyntheticInputTransport(BaseInputTransport):
    """Input transport that plays audio from bytes with real-time pacing.

    This transport is designed for benchmarking STT services in a Pipeline context.
    It reads pre-recorded audio and pushes it through the pipeline at real-time
    pace, simulating a real audio source as closely as possible.

    Plays audio for downstream processors:
    1. Streams all audio at real-time pace
    2. After audio ends, sends silence until transcription received (or timeout)
    3. Downstream VADProcessor handles speech start/end detection
    """

    def __init__(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        chunk_ms: int = 20,
        transcription_received: asyncio.Event | None = None,
        get_last_transcription_time: Callable[[], float] | None = None,
        max_silence_timeout: float = 10.0,
        post_transcription_delay: float = 2.0,
    ):
        """Initialize the synthetic input transport.

        Args:
            audio_data: Raw PCM audio bytes (16-bit, mono)
            sample_rate: Audio sample rate in Hz
            chunk_ms: Duration of each audio chunk in ms
            transcription_received: Event set when transcription is received (optional).
                If provided, silence is sent until this event is set or timeout.
                If None, sends a fixed amount of silence for compatibility.
            get_last_transcription_time: Callable returning the timestamp of the last
                received transcription frame. Used to ensure all trailing segments drain.
            max_silence_timeout: Maximum time to send silence in seconds (default 10.0s)
            post_transcription_delay: Time to continue sending silence after first
                transcription to collect additional segments (default 2.0s)
        """
        params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=sample_rate,
        )
        super().__init__(params)

        self._audio_data = audio_data
        self._sample_rate = sample_rate
        self._chunk_ms = chunk_ms
        self._transcription_received = transcription_received
        self._get_last_transcription_time = get_last_transcription_time
        self._max_silence_timeout = max_silence_timeout
        self._post_transcription_delay = post_transcription_delay

        # Calculate chunk size in bytes (16-bit audio = 2 bytes per sample)
        samples_per_chunk = int(sample_rate * chunk_ms / 1000)
        self._chunk_size = samples_per_chunk * 2

        # Total duration for logging
        total_samples = len(audio_data) // 2
        self._duration_seconds = total_samples / sample_rate

        # Pump task reference
        self._pump_task: asyncio.Task | None = None

        # Event to signal when audio pumping is complete
        self._audio_complete = asyncio.Event()

    @classmethod
    def from_file(
        cls,
        audio_path: str | Path,
        sample_rate: int = 16000,
        chunk_ms: int = 20,
        transcription_received: asyncio.Event | None = None,
        get_last_transcription_time: Callable[[], float] | None = None,
        max_silence_timeout: float = 10.0,
        post_transcription_delay: float = 2.0,
    ) -> "SyntheticInputTransport":
        """Create a transport from an audio file.

        Args:
            audio_path: Path to PCM audio file
            sample_rate: Audio sample rate in Hz
            chunk_ms: Duration of each audio chunk in ms
            transcription_received: Event set when transcription is received
            get_last_transcription_time: Callable returning the timestamp of the last
                received transcription frame.
            max_silence_timeout: Maximum time to send silence in seconds
            post_transcription_delay: Time to continue sending silence after first
                transcription to collect additional segments

        Returns:
            SyntheticInputTransport instance
        """
        audio_data = Path(audio_path).read_bytes()
        return cls(
            audio_data=audio_data,
            sample_rate=sample_rate,
            chunk_ms=chunk_ms,
            transcription_received=transcription_received,
            get_last_transcription_time=get_last_transcription_time,
            max_silence_timeout=max_silence_timeout,
            post_transcription_delay=post_transcription_delay,
        )

    async def start(self, frame: StartFrame):
        """Start the transport and begin pumping audio."""
        await super().start(frame)

        # Wait for transport to be ready
        await self.set_transport_ready(frame)

        logger.debug(
            f"SyntheticInputTransport starting: {self._duration_seconds:.2f}s audio, "
            f"{len(self._audio_data)} bytes, {self._chunk_ms}ms chunks"
        )

        # Launch the audio pumping task
        self._pump_task = self.create_task(self._pump_audio())

    async def _pump_audio(self):
        """Pump audio frames into the pipeline with real-time pacing."""
        try:
            sleep_time = self._chunk_ms / 1000
            silence_data = bytes(self._chunk_size)  # Zero-filled = silence

            # Send audio at real-time pace
            chunks_sent = 0
            for offset in range(0, len(self._audio_data), self._chunk_size):
                chunk = self._audio_data[offset : offset + self._chunk_size]

                frame = InputAudioRawFrame(
                    audio=chunk,
                    sample_rate=self._sample_rate,
                    num_channels=1,
                )
                await self.push_audio_frame(frame)
                chunks_sent += 1
                await asyncio.sleep(sleep_time)

            logger.debug(f"Sent {chunks_sent} audio chunks ({self._duration_seconds:.2f}s)")

            # Send silence until transcription received or timeout
            if self._transcription_received is not None:
                # Phase 1: Wait for first transcription
                silence_start = time.monotonic()
                silence_chunks_sent = 0

                while not self._transcription_received.is_set():
                    elapsed = time.monotonic() - silence_start
                    if elapsed >= self._max_silence_timeout:
                        logger.warning(
                            f"Max silence timeout ({self._max_silence_timeout}s) reached "
                            f"without transcription"
                        )
                        break
                    await self._send_silence_chunk(silence_data, sleep_time)
                    silence_chunks_sent += 1

                silence_duration = silence_chunks_sent * self._chunk_ms / 1000
                if self._transcription_received.is_set():
                    logger.debug(
                        f"First transcription confirmed (initial silence: {silence_duration:.2f}s)"
                    )

                    # Phase 2: keep sending silence until the service has gone
                    # quiet, rather than for a fixed span. It ends once
                    # post_transcription_delay has passed both since this phase
                    # began and since the last segment arrived, so a service
                    # still delivering segments keeps the window open. The
                    # max_silence_timeout below bounds the whole silence phase.
                    post_start = time.monotonic()
                    post_chunks = 0

                    drain_capped = False
                    while True:
                        now = time.monotonic()
                        last_tx = (
                            self._get_last_transcription_time()
                            if self._get_last_transcription_time
                            else post_start
                        )
                        if (now - post_start) >= self._post_transcription_delay and (
                            now - last_tx
                        ) >= self._post_transcription_delay:
                            break

                        if (now - silence_start) >= self._max_silence_timeout:
                            logger.warning(
                                f"Max silence timeout ({self._max_silence_timeout}s) reached "
                                f"while segments were still arriving; later ones are lost"
                            )
                            drain_capped = True
                            break

                        await self._send_silence_chunk(silence_data, sleep_time)
                        post_chunks += 1

                    post_duration = post_chunks * self._chunk_ms / 1000
                    ended = "cut off at the timeout" if drain_capped else "service went quiet"
                    logger.debug(
                        f"Post-transcription silence complete ({post_duration:.2f}s) - {ended}"
                    )
                else:
                    logger.debug(f"Silence phase ended after {silence_duration:.2f}s (timeout)")
            else:
                # Fallback: fixed silence for backwards compatibility
                for _ in range(5):  # 100ms at 20ms chunks
                    await self._send_silence_chunk(silence_data, sleep_time)
                logger.debug("Silence phase complete (fixed 100ms)")

            # Wait for audio queue to drain
            logger.debug("Waiting for audio queue to drain...")
            await self._audio_in_queue.join()
            logger.debug("Audio queue drained")

            # Signal completion
            logger.debug("Audio pumping complete")
            self._audio_complete.set()

        except asyncio.CancelledError:
            logger.debug("Audio pump task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in audio pump: {e}")
            self._audio_complete.set()

    async def _send_silence_chunk(self, silence_data: bytes, sleep_time: float):
        """Send a single silence chunk with real-time pacing."""
        silence_frame = InputAudioRawFrame(
            audio=silence_data,
            sample_rate=self._sample_rate,
            num_channels=1,
        )
        await self.push_audio_frame(silence_frame)
        await asyncio.sleep(sleep_time)

    async def wait_for_audio_complete(self, timeout: float = 60.0) -> bool:
        """Wait for audio pumping to complete.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if audio completed, False if timeout.
        """
        try:
            await asyncio.wait_for(self._audio_complete.wait(), timeout)
            return True
        except TimeoutError:
            return False

    @property
    def audio_complete(self) -> asyncio.Event:
        """Event that is set when audio pumping is complete."""
        return self._audio_complete

    async def cleanup(self):
        """Cleanup the transport."""
        if self._pump_task:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
            self._pump_task = None

        await super().cleanup()
