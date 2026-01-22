"""Observer for collecting transcription results from Pipecat pipeline."""

import asyncio

from loguru import logger
from pipecat.frames.frames import TranscriptionFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.services.stt_service import STTService


class TranscriptionCollectorObserver(BaseObserver):
    """Observer that captures final transcription results from STT services.

    Collects TranscriptionFrames (final results only) and concatenates them
    to build the complete transcription for WER calculation.
    """

    def __init__(self):
        super().__init__()
        # Final transcriptions keyed by sample ID (concatenated)
        self.transcriptions: dict[str, str] = {}

        # Current sample being processed
        self._current_sample_id: str | None = None

        # Event signaling that a final transcription was received
        self._transcription_received = asyncio.Event()

    def set_current_sample(self, sample_id: str) -> None:
        """Set the current sample being processed.

        Args:
            sample_id: The ID of the audio sample being processed.
        """
        self._current_sample_id = sample_id
        self._transcription_received.clear()
        # Clear any previous transcription for this sample (fresh start)
        if sample_id in self.transcriptions:
            del self.transcriptions[sample_id]

    def reset(self) -> None:
        """Reset the observer state for a new sample."""
        self._current_sample_id = None
        self._transcription_received.clear()

    async def on_push_frame(self, data: FramePushed) -> None:
        """Handle frame push events, capturing TranscriptionFrames only.

        Args:
            data: Frame push event data containing source, frame, and timestamp.
        """
        frame = data.frame

        # Only capture frames originating from the STT service
        if not isinstance(data.source, STTService):
            return

        if isinstance(frame, TranscriptionFrame):
            self._handle_transcription(frame.text)

    def _handle_transcription(self, text: str) -> None:
        """Handle a final transcription result by concatenating.

        Args:
            text: The transcription text segment.
        """
        if not self._current_sample_id:
            logger.warning("Received transcription but no current sample set")
            return

        # Concatenate final transcriptions (streaming STT sends multiple segments)
        if self._current_sample_id in self.transcriptions:
            self.transcriptions[self._current_sample_id] += " " + text
        else:
            self.transcriptions[self._current_sample_id] = text

        logger.debug(
            f"Transcription for {self._current_sample_id}: "
            f"'{text}' (total: {len(self.transcriptions[self._current_sample_id])} chars)"
        )

        # Signal that transcription was received
        self._transcription_received.set()

    async def wait_for_transcription(self, timeout: float = 30.0) -> str | None:
        """Wait for a final transcription to be received.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            The full concatenated transcription text, or None if timeout.
        """
        try:
            await asyncio.wait_for(self._transcription_received.wait(), timeout)
            return self.transcriptions.get(self._current_sample_id)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for transcription after {timeout}s")
            return None

    def get_transcription_for_sample(self, sample_id: str) -> str | None:
        """Get the full transcription for a specific sample.

        Args:
            sample_id: The sample ID to look up.

        Returns:
            The full concatenated transcription text, or None if not found.
        """
        return self.transcriptions.get(sample_id)

    def clear(self) -> None:
        """Clear all collected transcriptions."""
        self.transcriptions.clear()
        self._current_sample_id = None
        self._transcription_received.clear()
