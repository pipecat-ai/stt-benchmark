"""Observer for collecting TTFB metrics from Pipecat's MetricsFrame."""

import asyncio

from loguru import logger
from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.services.stt_service import STTService


class MetricsCollectorObserver(BaseObserver):
    """Observer that captures TTFB metrics from STT services.

    Pipecat's STTService automatically calculates TTFB (Time To First Byte)
    and emits MetricsFrame with TTFBMetricsData. This observer captures those
    metrics for benchmarking purposes.

    TTFB is measured from when the user stops speaking (VAD detection)
    to when the first TranscriptionFrame is received.
    """

    def __init__(self):
        super().__init__()
        # All TTFB values collected
        self.ttfb_values: list[float] = []

        # TTFB keyed by sample ID (when processing multiple samples)
        self.ttfb_by_sample: dict[str, float] = {}

        # Current sample being processed
        self._current_sample_id: str | None = None

        # Event signaling that a TTFB metric was received
        self._ttfb_received = asyncio.Event()

        # Latest TTFB value
        self._latest_ttfb: float | None = None

    def set_current_sample(self, sample_id: str) -> None:
        """Set the current sample being processed.

        Args:
            sample_id: The ID of the audio sample being processed.
        """
        self._current_sample_id = sample_id
        self._ttfb_received.clear()
        self._latest_ttfb = None

    def reset(self) -> None:
        """Reset the observer state for a new sample."""
        self._current_sample_id = None
        self._ttfb_received.clear()
        self._latest_ttfb = None

    async def on_push_frame(self, data: FramePushed) -> None:
        """Handle frame push events, capturing TTFB from MetricsFrame.

        Args:
            data: Frame push event data containing source, frame, and timestamp.
        """
        frame = data.frame

        # Only capture metrics originating from the STT service
        if not isinstance(data.source, STTService):
            return

        if not isinstance(frame, MetricsFrame):
            return

        # Process each metrics data item in the frame
        for metrics_data in frame.data:
            if isinstance(metrics_data, TTFBMetricsData):
                ttfb_value = metrics_data.value

                # Skip zero values (these are initialization metrics, not real TTFB)
                if ttfb_value == 0.0:
                    continue

                # Store the value
                self.ttfb_values.append(ttfb_value)
                self._latest_ttfb = ttfb_value

                # Associate with current sample if set
                if self._current_sample_id:
                    self.ttfb_by_sample[self._current_sample_id] = ttfb_value

                logger.debug(
                    f"Captured TTFB: {ttfb_value:.3f}s from {metrics_data.processor}"
                    f" (sample: {self._current_sample_id})"
                )

                # Signal that TTFB was received
                self._ttfb_received.set()

    async def wait_for_ttfb(self, timeout: float = 30.0) -> float | None:
        """Wait for a TTFB metric to be received.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            The TTFB value in seconds, or None if timeout.
        """
        try:
            await asyncio.wait_for(self._ttfb_received.wait(), timeout)
            return self._latest_ttfb
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for TTFB metric after {timeout}s")
            return None

    def get_ttfb_for_sample(self, sample_id: str) -> float | None:
        """Get the TTFB value for a specific sample.

        Args:
            sample_id: The sample ID to look up.

        Returns:
            The TTFB value in seconds, or None if not found.
        """
        return self.ttfb_by_sample.get(sample_id)

    def clear(self) -> None:
        """Clear all collected metrics."""
        self.ttfb_values.clear()
        self.ttfb_by_sample.clear()
        self._current_sample_id = None
        self._ttfb_received.clear()
        self._latest_ttfb = None
