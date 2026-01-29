"""STT Service configurations using factory functions.

Each service is defined as a factory function that returns a configured
Pipecat STT service instance. This gives full control over constructor
arguments for each service.

To add a new service or modify an existing one:
1. Create/modify a factory function that returns the configured service
2. Add/update the entry in STT_SERVICES with required env vars

Example - modifying Gradium to use a US endpoint:

    def create_gradium() -> FrameProcessor:
        from pipecat.services.gradium.stt import GradiumSTTService
        return GradiumSTTService(
            api_key=_get_env("GRADIUM_API_KEY"),
            api_endpoint_base_url="wss://us.api.gradium.ai/api/speech/asr",
        )
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transcriptions.language import Language

if TYPE_CHECKING:
    from stt_benchmark.models import ServiceName


def _get_env(name: str) -> str:
    """Get environment variable from config (supports .env files), raising if not set."""
    from stt_benchmark.config import get_config

    config = get_config()
    attr_name = name.lower()
    # Try to get from config first (which loads .env)
    value = getattr(config, attr_name, None)
    if value:
        return value
    # Fall back to os.getenv for env vars not in config
    value = os.getenv(name, "")
    if not value:
        raise ValueError(f"{name} environment variable not set")
    return value


# Type alias for service factory functions
ServiceFactory = Callable[[], FrameProcessor]


@dataclass
class ServiceDefinition:
    """Definition of an STT service."""

    # Factory function that creates the configured service instance
    factory: ServiceFactory

    # Environment variables required for this service
    # Used to check if service is available before attempting to create it
    required_env_vars: list[str] = field(default_factory=list)


# =============================================================================
# SERVICE FACTORY FUNCTIONS
# =============================================================================
# Each factory returns a fully configured Pipecat STT service instance.
# Modify these to change service configuration (models, endpoints, params, etc.)
# =============================================================================


def create_assemblyai() -> FrameProcessor:
    from pipecat.services.assemblyai.stt import AssemblyAISTTService

    return AssemblyAISTTService(
        api_key=_get_env("ASSEMBLYAI_API_KEY"),
    )


def create_aws() -> FrameProcessor:
    from pipecat.services.aws.stt import AWSTranscribeSTTService

    return AWSTranscribeSTTService(
        api_key=_get_env("AWS_SECRET_ACCESS_KEY"),
        aws_access_key_id=_get_env("AWS_ACCESS_KEY_ID"),
        region=_get_env("AWS_REGION"),
    )


def create_azure() -> FrameProcessor:
    from pipecat.services.azure.stt import AzureSTTService

    return AzureSTTService(
        api_key=_get_env("AZURE_SPEECH_API_KEY"),
        region=_get_env("AZURE_SPEECH_REGION"),
    )


def create_cartesia() -> FrameProcessor:
    from pipecat.services.cartesia.stt import CartesiaSTTService

    return CartesiaSTTService(
        api_key=_get_env("CARTESIA_API_KEY"),
        model="ink-whisper",
    )


def create_deepgram() -> FrameProcessor:
    from deepgram import LiveOptions
    from pipecat.services.deepgram.stt import DeepgramSTTService

    return DeepgramSTTService(
        api_key=_get_env("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            model="nova-3-general",
            language=Language.EN,
        ),
    )


def create_elevenlabs() -> FrameProcessor:
    from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService

    return ElevenLabsRealtimeSTTService(
        api_key=_get_env("ELEVENLABS_API_KEY"),
        model="scribe_v2_realtime",
        params=ElevenLabsRealtimeSTTService.InputParams(
            language=Language.EN,
        ),
    )


def create_fal() -> FrameProcessor:
    from pipecat.services.fal.stt import FalSTTService

    return FalSTTService(
        api_key=_get_env("FAL_KEY"),
        params=FalSTTService.InputParams(
            language=Language.EN,
        ),
    )


def create_gladia() -> FrameProcessor:
    from pipecat.services.gladia.config import GladiaInputParams, LanguageConfig
    from pipecat.services.gladia.stt import GladiaSTTService

    return GladiaSTTService(
        api_key=_get_env("GLADIA_API_KEY"),
        region=os.getenv("GLADIA_REGION", "us-west"),
        model="solaria-1",
        params=GladiaInputParams(
            language_config=LanguageConfig(
                languages=[Language.EN],
            )
        ),
    )


def create_google() -> FrameProcessor:
    from pipecat.services.google.stt import GoogleSTTService

    return GoogleSTTService(
        credentials_path=_get_env("GOOGLE_APPLICATION_CREDENTIALS"),
        location=os.getenv("GOOGLE_LOCATION", "us-central1"),
        params=GoogleSTTService.InputParams(
            languages=Language.EN_US,
            model="latest_long",
        ),
    )


def create_gradium() -> FrameProcessor:
    from pipecat.services.gradium.stt import GradiumSTTService

    return GradiumSTTService(
        api_key=_get_env("GRADIUM_API_KEY"),
        api_endpoint_base_url=os.getenv(
            "GRADIUM_BASE_URL", "wss://us.api.gradium.ai/api/speech/asr"
        ),
        params=GradiumSTTService.InputParams(
            language=Language.EN,
        ),
    )


def create_groq() -> FrameProcessor:
    from pipecat.services.groq.stt import GroqSTTService

    return GroqSTTService(
        api_key=_get_env("GROQ_API_KEY"),
        model="whisper-large-v3-turbo",
        language=Language.EN,
    )


def create_nvidia() -> FrameProcessor:
    from pipecat.services.nvidia.stt import NvidiaSTTService

    return NvidiaSTTService(
        api_key=_get_env("NVIDIA_API_KEY"),
        params=NvidiaSTTService.InputParams(
            language=Language.EN_US,
        ),
    )


def create_openai() -> FrameProcessor:
    from pipecat.services.openai.stt import OpenAISTTService

    return OpenAISTTService(
        api_key=_get_env("OPENAI_API_KEY"),
        model="gpt-4o-mini-transcribe",
        language=Language.EN,
    )


def create_sambanova() -> FrameProcessor:
    from pipecat.services.sambanova.stt import SambaNovaSTTService

    return SambaNovaSTTService(
        api_key=_get_env("SAMBANOVA_API_KEY"),
        model="Whisper-Large-v3",
        language=Language.EN,
    )


def create_sarvam() -> FrameProcessor:
    from pipecat.services.sarvam.stt import SarvamSTTService

    return SarvamSTTService(
        api_key=_get_env("SARVAM_API_KEY"),
        model="saarika:v2.5",
    )


def create_soniox() -> FrameProcessor:
    from pipecat.services.soniox.stt import SonioxInputParams, SonioxSTTService

    return SonioxSTTService(
        api_key=_get_env("SONIOX_API_KEY"),
        params=SonioxInputParams(
            model="stt-rt-preview",
            language_hints=[Language.EN],
            language_hints_strict=True,
        ),
        vad_force_turn_endpoint=True,
    )


def create_speechmatics() -> FrameProcessor:
    from pipecat.services.speechmatics.stt import SpeechmaticsSTTService, TurnDetectionMode

    return SpeechmaticsSTTService(
        api_key=_get_env("SPEECHMATICS_API_KEY"),
        base_url=os.getenv("SPEECHMATICS_RT_URL", "wss://us.rt.speechmatics.com/v2"),
        params=SpeechmaticsSTTService.InputParams(
            language=Language.EN,
            turn_detection_mode=TurnDetectionMode.EXTERNAL,
        ),
    )


def create_whisper() -> FrameProcessor:
    from pipecat.services.whisper.stt import Model, WhisperSTTService

    return WhisperSTTService(
        model=Model.DISTIL_MEDIUM_EN,
        language=Language.EN,
    )


# =============================================================================
# SERVICE REGISTRY
# =============================================================================
# Maps service names to their definitions (factory + required env vars).
# The required_env_vars are used to check availability before creating.
# =============================================================================

STT_SERVICES: dict[str, ServiceDefinition] = {
    "assemblyai": ServiceDefinition(
        factory=create_assemblyai,
        required_env_vars=["ASSEMBLYAI_API_KEY"],
    ),
    "aws": ServiceDefinition(
        factory=create_aws,
        required_env_vars=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
    ),
    "azure": ServiceDefinition(
        factory=create_azure,
        required_env_vars=["AZURE_SPEECH_API_KEY", "AZURE_SPEECH_REGION"],
    ),
    "cartesia": ServiceDefinition(
        factory=create_cartesia,
        required_env_vars=["CARTESIA_API_KEY"],
    ),
    "deepgram": ServiceDefinition(
        factory=create_deepgram,
        required_env_vars=["DEEPGRAM_API_KEY"],
    ),
    "elevenlabs": ServiceDefinition(
        factory=create_elevenlabs,
        required_env_vars=["ELEVENLABS_API_KEY"],
    ),
    "fal": ServiceDefinition(
        factory=create_fal,
        required_env_vars=["FAL_KEY"],
    ),
    "gladia": ServiceDefinition(
        factory=create_gladia,
        required_env_vars=["GLADIA_API_KEY"],
    ),
    "google": ServiceDefinition(
        factory=create_google,
        required_env_vars=["GOOGLE_APPLICATION_CREDENTIALS"],
    ),
    "gradium": ServiceDefinition(
        factory=create_gradium,
        required_env_vars=["GRADIUM_API_KEY"],
    ),
    "groq": ServiceDefinition(
        factory=create_groq,
        required_env_vars=["GROQ_API_KEY"],
    ),
    "nvidia": ServiceDefinition(
        factory=create_nvidia,
        required_env_vars=["NVIDIA_API_KEY"],
    ),
    "openai": ServiceDefinition(
        factory=create_openai,
        required_env_vars=["OPENAI_API_KEY"],
    ),
    "sambanova": ServiceDefinition(
        factory=create_sambanova,
        required_env_vars=["SAMBANOVA_API_KEY"],
    ),
    "sarvam": ServiceDefinition(
        factory=create_sarvam,
        required_env_vars=["SARVAM_API_KEY"],
    ),
    "soniox": ServiceDefinition(
        factory=create_soniox,
        required_env_vars=["SONIOX_API_KEY"],
    ),
    "speechmatics": ServiceDefinition(
        factory=create_speechmatics,
        required_env_vars=["SPEECHMATICS_API_KEY"],
    ),
    "whisper": ServiceDefinition(
        factory=create_whisper,
        required_env_vars=[],  # Local model, no API key needed
    ),
}


# =============================================================================
# SERVICE CREATION & AVAILABILITY
# =============================================================================


def get_service_definition(name: str) -> ServiceDefinition:
    """Get the service definition by name."""
    if name not in STT_SERVICES:
        raise ValueError(f"Unknown service: {name}. Available: {list(STT_SERVICES.keys())}")
    return STT_SERVICES[name]


def get_all_service_names() -> list[str]:
    """Get all configured service names."""
    return list(STT_SERVICES.keys())


def _get_env_from_config(env_var_name: str) -> str:
    """Get environment variable value from config (supports .env files via Pydantic).

    Derives config attribute from env var name: DEEPGRAM_API_KEY -> deepgram_api_key
    Falls back to os.getenv() for env vars not in config.
    """
    from stt_benchmark.config import get_config

    config = get_config()
    attr_name = env_var_name.lower()
    # Try to get from config first (which loads .env)
    value = getattr(config, attr_name, None)
    if value is not None:
        return value
    # Fall back to os.getenv for env vars not in config
    return os.getenv(env_var_name, "")


def is_service_available(name: str) -> bool:
    """Check if a service has all required environment variables set."""
    if name not in STT_SERVICES:
        return False
    definition = STT_SERVICES[name]
    return all(_get_env_from_config(env_var) for env_var in definition.required_env_vars)


def create_stt_service(service_name: "ServiceName") -> FrameProcessor:
    """Create an STT service instance using its factory function.

    Args:
        service_name: The STT service to create.

    Returns:
        Configured STT service instance.

    Raises:
        ValueError: If service_name is not supported or required credentials are missing.
    """
    from loguru import logger

    definition = get_service_definition(service_name.value)
    logger.debug(f"Creating {service_name.value} STT service")
    return definition.factory()


def get_available_services() -> list["ServiceName"]:
    """Get list of services that have all required credentials configured.

    Returns:
        List of ServiceName values for available services.
    """
    from loguru import logger

    from stt_benchmark.models import ServiceName

    available = []
    for name in STT_SERVICES:
        if is_service_available(name):
            try:
                available.append(ServiceName(name))
            except ValueError:
                logger.warning(f"Service {name} not in ServiceName enum")
        else:
            definition = STT_SERVICES[name]
            logger.debug(
                f"Service {name} not available (missing env vars: {definition.required_env_vars})"
            )
    return available


def get_all_services() -> list["ServiceName"]:
    """Get list of all supported services.

    Returns:
        List of all ServiceName values.
    """
    from stt_benchmark.models import ServiceName

    return list(ServiceName)


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
    if services_arg.lower() == "all":
        return get_available_services()

    return [parse_service_name(s) for s in services_arg.split(",")]
