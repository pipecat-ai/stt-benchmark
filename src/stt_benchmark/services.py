"""STT Service configurations.

This file defines all STT services and their configurations in one place,
making it easy to see and modify what's being tested.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stt_benchmark.models import ServiceName


@dataclass
class STTServiceConfig:
    """Configuration for an STT service."""

    # Service identifier (matches ServiceName enum)
    name: str

    # Pipecat service class (full import path)
    pipecat_module: str
    pipecat_class: str

    # Environment variable for API key (simple case)
    # Maps to constructor param "api_key"
    api_key_env: str | None = None

    # Environment variables mapping: param_name -> ENV_VAR_NAME
    # Use this for services needing multiple credentials (AWS, Azure, etc.)
    # Example: {"aws_access_key_id": "AWS_ACCESS_KEY_ID", "region": "AWS_REGION"}
    env_vars: dict[str, str] = field(default_factory=dict)

    # Default model to use
    default_model: str | None = None

    # Whether service requires aiohttp session
    needs_aiohttp: bool = False

    # Additional kwargs to pass to service constructor
    extra_kwargs: dict = field(default_factory=dict)


# =============================================================================
# STT SERVICE CONFIGURATIONS
# =============================================================================
# Add, remove, or modify services here. Each service needs:
# - name: identifier used in CLI and database
# - api_key_env: environment variable name for API key
# - pipecat_module/class: Pipecat service import path
# - default_model: model to use if none specified
# =============================================================================

STT_SERVICES: dict[str, STTServiceConfig] = {
    "assemblyai": STTServiceConfig(
        name="assemblyai",
        api_key_env="ASSEMBLYAI_API_KEY",
        pipecat_module="pipecat.services.assemblyai.stt",
        pipecat_class="AssemblyAISTTService",
        default_model=None,
    ),
    "aws": STTServiceConfig(
        name="aws",
        pipecat_module="pipecat.services.aws.stt",
        pipecat_class="AWSTranscribeSTTService",
        env_vars={
            "aws_access_key_id": "AWS_ACCESS_KEY_ID",
            "api_key": "AWS_SECRET_ACCESS_KEY",
            "region": "AWS_REGION",
        },
        default_model=None,
    ),
    "azure": STTServiceConfig(
        name="azure",
        pipecat_module="pipecat.services.azure.stt",
        pipecat_class="AzureSTTService",
        env_vars={
            "api_key": "AZURE_SPEECH_API_KEY",
            "region": "AZURE_SPEECH_REGION",
        },
        default_model=None,
    ),
    "cartesia": STTServiceConfig(
        name="cartesia",
        api_key_env="CARTESIA_API_KEY",
        pipecat_module="pipecat.services.cartesia.stt",
        pipecat_class="CartesiaSTTService",
        default_model="ink-whisper",
    ),
    "deepgram": STTServiceConfig(
        name="deepgram",
        api_key_env="DEEPGRAM_API_KEY",
        pipecat_module="pipecat.services.deepgram.stt",
        pipecat_class="DeepgramSTTService",
        default_model="nova-3-general",
    ),
    # "deepgram_flux": STTServiceConfig(
    #     name="deepgram_flux",
    #     api_key_env="DEEPGRAM_API_KEY",
    #     pipecat_module="pipecat.services.deepgram.flux.stt",
    #     pipecat_class="DeepgramFluxSTTService",
    #     default_model="flux-general-en",
    # ),
    "elevenlabs": STTServiceConfig(
        name="elevenlabs",
        api_key_env="ELEVENLABS_API_KEY",
        pipecat_module="pipecat.services.elevenlabs.stt",
        pipecat_class="ElevenLabsRealtimeSTTService",
        default_model="scribe_v2_realtime",
    ),
    "fal": STTServiceConfig(
        name="fal",
        api_key_env="FAL_KEY",
        pipecat_module="pipecat.services.fal.stt",
        pipecat_class="FalSTTService",
        default_model=None,
    ),
    "gladia": STTServiceConfig(
        name="gladia",
        api_key_env="GLADIA_API_KEY",
        pipecat_module="pipecat.services.gladia.stt",
        pipecat_class="GladiaSTTService",
        default_model="solaria-1",
    ),
    "google": STTServiceConfig(
        name="google",
        pipecat_module="pipecat.services.google.stt",
        pipecat_class="GoogleSTTService",
        env_vars={
            "credentials_path": "GOOGLE_APPLICATION_CREDENTIALS",
        },
        default_model=None,
    ),
    "gradium": STTServiceConfig(
        name="gradium",
        api_key_env="GRADIUM_API_KEY",
        pipecat_module="pipecat.services.gradium.stt",
        pipecat_class="GradiumSTTService",
        default_model=None,
    ),
    "groq": STTServiceConfig(
        name="groq",
        api_key_env="GROQ_API_KEY",
        pipecat_module="pipecat.services.groq.stt",
        pipecat_class="GroqSTTService",
        default_model="whisper-large-v3-turbo",
    ),
    "hathora": STTServiceConfig(
        name="hathora",
        api_key_env="HATHORA_API_KEY",
        pipecat_module="pipecat.services.hathora.stt",
        pipecat_class="HathoraSTTService",
        default_model="nvidia-parakeet-tdt-0.6b-v3",
    ),
    "nvidia": STTServiceConfig(
        name="nvidia",
        api_key_env="NVIDIA_API_KEY",
        pipecat_module="pipecat.services.nvidia.stt",
        pipecat_class="NvidiaSTTService",
        default_model=None,
    ),
    "openai": STTServiceConfig(
        name="openai",
        api_key_env="OPENAI_API_KEY",
        pipecat_module="pipecat.services.openai.stt",
        pipecat_class="OpenAISTTService",
        default_model="gpt-4o-transcribe",
    ),
    "sambanova": STTServiceConfig(
        name="sambanova",
        api_key_env="SAMBANOVA_API_KEY",
        pipecat_module="pipecat.services.sambanova.stt",
        pipecat_class="SambaNovaSTTService",
        default_model="Whisper-Large-v3",
    ),
    "sarvam": STTServiceConfig(
        name="sarvam",
        api_key_env="SARVAM_API_KEY",
        pipecat_module="pipecat.services.sarvam.stt",
        pipecat_class="SarvamSTTService",
        default_model="saarika:v2.5",
    ),
    "soniox": STTServiceConfig(
        name="soniox",
        api_key_env="SONIOX_API_KEY",
        pipecat_module="pipecat.services.soniox.stt",
        pipecat_class="SonioxSTTService",
        default_model=None,
    ),
    "speechmatics": STTServiceConfig(
        name="speechmatics",
        api_key_env="SPEECHMATICS_API_KEY",
        pipecat_module="pipecat.services.speechmatics.stt",
        pipecat_class="SpeechmaticsSTTService",
        default_model=None,
    ),
    "whisper": STTServiceConfig(
        name="whisper",
        pipecat_module="pipecat.services.whisper.stt",
        pipecat_class="WhisperSTTService",
        default_model="Systran/faster-distil-whisper-medium.en",
    ),
}


def get_service_config(name: str) -> STTServiceConfig:
    """Get configuration for a service by name."""
    if name not in STT_SERVICES:
        raise ValueError(f"Unknown service: {name}. Available: {list(STT_SERVICES.keys())}")
    return STT_SERVICES[name]


def get_all_service_names() -> list[str]:
    """Get all configured service names."""
    return list(STT_SERVICES.keys())


# =============================================================================
# CLI UTILITIES
# =============================================================================


def parse_service_name(name: str) -> "ServiceName":
    """Parse a service name string to ServiceName enum.

    Args:
        name: Service name (case-insensitive)

    Returns:
        ServiceName enum value

    Raises:
        ValueError: If service name is unknown
    """
    from stt_benchmark.models import ServiceName

    name_lower = name.strip().lower()
    if name_lower not in STT_SERVICES:
        raise ValueError(f"Unknown service: {name}. Available: {', '.join(STT_SERVICES.keys())}")
    return ServiceName(name_lower)


def parse_services_arg(services_arg: str) -> list["ServiceName"]:
    """Parse a comma-separated services argument.

    Args:
        services_arg: Comma-separated service names or 'all'

    Returns:
        List of ServiceName enum values
    """
    from stt_benchmark.pipeline.service_factory import get_available_services

    if services_arg.lower() == "all":
        return get_available_services()

    return [parse_service_name(s) for s in services_arg.split(",")]
