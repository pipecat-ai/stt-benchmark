"""Configuration management for STT benchmarking."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BenchmarkConfig(BaseSettings):
    """Configuration for STT benchmarking."""

    model_config = SettingsConfigDict(
        env_prefix="STT_BENCHMARK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Field(default=Path("./stt_benchmark_data"))
    audio_dir: Path = Field(default=Path("./stt_benchmark_data/audio"))
    results_db: Path = Field(default=Path("./stt_benchmark_data/results.db"))

    # Dataset configuration
    dataset_name: str = "pipecat-ai/smart-turn-data-v3.1-train"
    num_samples: int = 100
    seed: int = 42

    # API Keys (loaded from environment)
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    assemblyai_api_key: str = Field(default="", alias="ASSEMBLYAI_API_KEY")
    cartesia_api_key: str = Field(default="", alias="CARTESIA_API_KEY")
    deepgram_api_key: str = Field(default="", alias="DEEPGRAM_API_KEY")
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")
    fal_key: str = Field(default="", alias="FAL_KEY")
    gladia_api_key: str = Field(default="", alias="GLADIA_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    google_application_credentials: str = Field(default="", alias="GOOGLE_APPLICATION_CREDENTIALS")
    gradium_api_key: str = Field(default="", alias="GRADIUM_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    hathora_api_key: str = Field(default="", alias="HATHORA_API_KEY")
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    sambanova_api_key: str = Field(default="", alias="SAMBANOVA_API_KEY")
    sarvam_api_key: str = Field(default="", alias="SARVAM_API_KEY")
    soniox_api_key: str = Field(default="", alias="SONIOX_API_KEY")
    speechmatics_api_key: str = Field(default="", alias="SPEECHMATICS_API_KEY")

    # AWS credentials
    aws_access_key_id: str = Field(default="", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="", alias="AWS_REGION")

    # Azure credentials
    azure_speech_api_key: str = Field(default="", alias="AZURE_SPEECH_API_KEY")
    azure_speech_region: str = Field(default="", alias="AZURE_SPEECH_REGION")

    # Gemini rate limiting
    gemini_requests_per_minute: int = 60

    # Audio configuration
    sample_rate: int = 16000
    chunk_duration_ms: int = 20

    # VAD configuration
    vad_stop_secs: float = 0.2

    # Benchmark settings
    max_silence_timeout_secs: float = (
        10.0  # Max time to send silence while waiting for transcription
    )
    transcription_timeout_secs: float = (
        10.0  # Max time to wait for transcription after silence ends
    )

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    @property
    def chunk_size_bytes(self) -> int:
        """Calculate chunk size in bytes for streaming."""
        samples_per_chunk = int(self.sample_rate * self.chunk_duration_ms / 1000)
        return samples_per_chunk * 2  # 16-bit audio = 2 bytes per sample


# Global configuration instance
_config: BenchmarkConfig | None = None


def get_config() -> BenchmarkConfig:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = BenchmarkConfig()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
