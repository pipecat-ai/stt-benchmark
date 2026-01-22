"""Data models for STT benchmarking."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


class ServiceName(str, Enum):
    """Supported STT services."""

    ASSEMBLYAI = "assemblyai"
    AWS = "aws"
    AZURE = "azure"
    CARTESIA = "cartesia"
    DEEPGRAM = "deepgram"
    # DEEPGRAM_FLUX = "deepgram_flux"
    ELEVENLABS = "elevenlabs"
    FAL = "fal"
    GLADIA = "gladia"
    GOOGLE = "google"
    GRADIUM = "gradium"
    GROQ = "groq"
    HATHORA = "hathora"
    NVIDIA = "nvidia"
    OPENAI = "openai"
    SAMBANOVA = "sambanova"
    SARVAM = "sarvam"
    SPEECHMATICS = "speechmatics"
    SONIOX = "soniox"
    WHISPER = "whisper"


class AudioSample(BaseModel):
    """A single audio sample from the dataset."""

    sample_id: str = Field(description="Unique identifier for the sample")
    audio_path: str = Field(description="Local path to the PCM audio file")
    duration_seconds: float = Field(description="Audio duration in seconds")
    language: str = Field(default="eng", description="Language code")
    dataset_index: int = Field(description="Original index in HuggingFace dataset")


class BenchmarkResult(BaseModel):
    """Result from a single STT benchmark run."""

    sample_id: str = Field(description="Reference to AudioSample.sample_id")
    service_name: ServiceName = Field(description="STT service used")
    model_name: str | None = Field(default=None, description="Model name if applicable")

    # TTFB metrics (from Pipecat's MetricsFrame)
    ttfb_seconds: float | None = Field(
        default=None, description="Time to first byte in seconds (from MetricsFrame)"
    )

    # Transcription result
    transcription: str | None = Field(default=None, description="Final transcription text")

    # Audio metadata
    audio_duration_seconds: float = Field(description="Duration of the audio sample")

    # Timing
    timestamp: datetime = Field(default_factory=_utcnow)

    # Error tracking
    error: str | None = Field(default=None, description="Error message if failed")


class AggregateStatistics(BaseModel):
    """Aggregate statistics for a service."""

    service_name: ServiceName
    model_name: str | None = None
    num_samples: int
    num_errors: int = Field(description="Number of samples with errors")

    # TTFB statistics (in seconds)
    ttfb_mean: float | None = None
    ttfb_median: float | None = None
    ttfb_std: float | None = None
    ttfb_min: float | None = None
    ttfb_max: float | None = None
    ttfb_p50: float | None = None
    ttfb_p90: float | None = None
    ttfb_p95: float | None = None
    ttfb_p99: float | None = None

    # TTFB by audio duration bucket
    ttfb_by_duration: dict[str, float] | None = Field(
        default=None, description="Mean TTFB by audio duration bucket (e.g., '0-2s', '2-5s')"
    )


class BenchmarkRun(BaseModel):
    """Metadata for a benchmark run."""

    run_id: str = Field(description="Unique identifier for the run")
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    services: list[ServiceName] = Field(description="Services benchmarked")
    num_samples: int = Field(description="Number of samples processed")
    config_snapshot: dict | None = Field(default=None, description="Configuration at time of run")


class GroundTruth(BaseModel):
    """Ground truth transcription for an audio sample."""

    sample_id: str = Field(description="Reference to AudioSample.sample_id")
    text: str = Field(description="Ground truth transcription text")
    model_used: str = Field(
        default="gemini-3-flash-preview", description="Model used for transcription"
    )
    generated_at: datetime = Field(default_factory=_utcnow)


class SemanticError(BaseModel):
    """A semantically meaningful transcription error."""

    error_type: str = Field(description="Type of error: substitution, deletion, insertion")
    reference_word: str | None = Field(
        default=None, description="Word from ground truth (for substitution/deletion)"
    )
    hypothesis_word: str | None = Field(
        default=None, description="Word from transcription (for substitution/insertion)"
    )
    position: int | None = Field(default=None, description="Position in alignment")


class WERMetrics(BaseModel):
    """Semantic Word Error Rate metrics for a transcription.

    Uses Claude to evaluate only errors that would impact how an LLM agent
    understands and responds to the user.
    """

    sample_id: str = Field(description="Reference to AudioSample.sample_id")
    service_name: ServiceName = Field(description="STT service evaluated")
    model_name: str | None = Field(default=None, description="Model name if applicable")

    # Semantic WER metrics (from Claude evaluation)
    wer: float = Field(description="Semantic Word Error Rate (0-1+)")
    substitutions: int = Field(description="Number of semantic substitutions")
    deletions: int = Field(description="Number of semantic deletions")
    insertions: int = Field(description="Number of semantic insertions")
    reference_words: int = Field(description="Total words in normalized reference")

    # Semantic error details
    errors: list[SemanticError] | None = Field(
        default=None, description="List of identified semantic errors"
    )

    # Normalized texts (as determined by Claude)
    normalized_reference: str | None = Field(
        default=None, description="Claude-normalized reference text"
    )
    normalized_hypothesis: str | None = Field(
        default=None, description="Claude-normalized hypothesis text"
    )

    timestamp: datetime = Field(default_factory=_utcnow)


class SemanticWERTrace(BaseModel):
    """Full reasoning trace from Claude semantic WER evaluation."""

    sample_id: str = Field(description="Reference to AudioSample.sample_id")
    service_name: ServiceName = Field(description="STT service evaluated")
    model_name: str | None = Field(default=None, description="Model name if applicable")
    session_id: str = Field(description="Unique session identifier")

    # Full conversation trace
    conversation_trace: list[dict] = Field(description="Full message history from the evaluation")
    tool_calls: list[dict] = Field(description="All tool invocations during evaluation")

    # Normalized texts (as determined by Claude)
    normalized_reference: str | None = Field(
        default=None, description="Claude-normalized reference text"
    )
    normalized_hypothesis: str | None = Field(
        default=None, description="Claude-normalized hypothesis text"
    )

    # WER calculation results
    wer: float = Field(description="Calculated semantic WER")
    substitutions: int = Field(description="Number of substitutions")
    deletions: int = Field(description="Number of deletions")
    insertions: int = Field(description="Number of insertions")
    reference_words: int = Field(description="Total reference words after normalization")
    errors: list[SemanticError] | None = Field(
        default=None, description="Identified semantic errors"
    )

    # Performance metrics
    duration_ms: int | None = Field(default=None, description="Total evaluation time")
    num_turns: int = Field(default=1, description="Number of conversation turns")
    model_used: str = Field(
        default="claude-sonnet-4-5-20250929", description="Model used for evaluation"
    )

    timestamp: datetime = Field(default_factory=_utcnow)


class AggregateWERStatistics(BaseModel):
    """Aggregate semantic WER statistics for a service."""

    service_name: ServiceName
    model_name: str | None = None
    num_samples: int

    # Semantic WER statistics
    wer_mean: float | None = None
    wer_median: float | None = None
    wer_std: float | None = None
    wer_min: float | None = None
    wer_max: float | None = None

    # Pooled WER (sum of errors / sum of reference words)
    pooled_wer: float | None = None
