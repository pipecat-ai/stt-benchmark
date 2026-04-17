"""Ground truth generation for WER calculation."""

from stt_benchmark.ground_truth.gemini_transcriber import GeminiTranscriber
from stt_benchmark.ground_truth.scribe_transcriber import ScribeTranscriber

__all__ = ["GeminiTranscriber", "ScribeTranscriber"]
