"""STT Service configurations using factory functions.

Each service is defined as a factory function that returns a configured
Pipecat STT service instance. This gives full control over constructor
arguments for each service.

One entry == one (vendor, model). When a vendor ships a new model, ADD a new
entry rather than editing the existing one in place — that keeps the old model's
published numbers reproducible and lets the benchmark show the before/after.
A vendor's first/original entry keeps the bare vendor key (e.g. ``assemblyai``,
``cartesia``). Going forward, every NEW model uses a full ``vendor_model`` key
derived from the model string (e.g. ``assemblyai_u3_rt_pro`` for model
``u3-rt-pro``), so the key is unambiguous on its own. Older keys
(``cartesia_ink2``) are not renamed. Mark the superseded entry
``is_current=False``. Full checklist: docs/adding-models.md.

To add or update a service:
1. Create a factory function that returns the configured service
2. Add a ServiceName enum value in models.py
3. Add the entry in STT_SERVICES with vendor, model_label, and required env vars

Example - modifying Gradium to use a US endpoint:

    def create_gradium() -> FrameProcessor:
        from pipecat.services.gradium.stt import GradiumSTTService
        return GradiumSTTService(
            api_key=_get_env("GRADIUM_API_KEY"),
            api_endpoint_base_url="wss://us.api.gradium.ai/api/speech/asr",
        )
"""

import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transcriptions.language import Language

if TYPE_CHECKING:
    import aiohttp

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
    """Definition of an STT service.

    A "service" is one (vendor, model) pair. A vendor with multiple models has
    one entry per model (e.g. ``cartesia`` + ``cartesia_ink2``). ``vendor`` and
    ``model_label`` make each entry self-describing so results carry a
    human-readable vendor/model label without consulting a separate map; set
    ``is_current=False`` on a model that a newer entry has superseded.
    See docs/adding-models.md.
    """

    # Factory function that creates the configured service instance
    factory: ServiceFactory

    # Human-readable vendor name shared across that vendor's models (e.g. "AssemblyAI")
    vendor: str

    # Human-readable label for this specific model (e.g. "Universal-3 RT Pro")
    model_label: str

    # Environment variables required for this service
    # Used to check if service is available before attempting to create it
    required_env_vars: list[str] = field(default_factory=list)

    # Whether this service requires an aiohttp.ClientSession to be passed
    # to the factory. When True, the pipeline runner will create a session
    # context and pass it as the first argument to the factory.
    needs_aiohttp: bool = False

    # False when a newer model entry from the same vendor supersedes this one.
    # Superseded models stay runnable (for reproducibility) but are not the
    # vendor's headline/recommended model.
    is_current: bool = True


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
        settings=AssemblyAISTTService.Settings(
            model="universal-streaming-english",
            end_of_turn_confidence_threshold=1.0,
            max_turn_silence=2000,
        ),
        vad_force_turn_endpoint=True,
    )


def create_assemblyai_u3_rt_pro() -> FrameProcessor:
    from pipecat.services.assemblyai.stt import AssemblyAISTTService

    return AssemblyAISTTService(
        api_key=_get_env("ASSEMBLYAI_API_KEY"),
        settings=AssemblyAISTTService.Settings(
            model="u3-rt-pro",
            end_of_turn_confidence_threshold=1.0,
            min_turn_silence=50,
            max_turn_silence=50,
            vad_threshold=0.2,
        ),
        vad_force_turn_endpoint=True,
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
        settings=CartesiaSTTService.Settings(
            model="ink-whisper",
            language=Language.EN,
        ),
    )


def create_cartesia_ink2() -> FrameProcessor:
    from pipecat.services.cartesia.stt import CartesiaSTTService

    return CartesiaSTTService(
        api_key=_get_env("CARTESIA_API_KEY"),
        settings=CartesiaSTTService.Settings(
            model="ink-2",
            language=Language.EN,
        ),
    )


def create_deepgram() -> FrameProcessor:
    from pipecat.services.deepgram.stt import DeepgramSTTService

    return DeepgramSTTService(
        api_key=_get_env("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model="nova-3-general",
            smart_format=False,
            profanity_filter=False,
            language=Language.EN,
        ),
    )


def create_elevenlabs() -> FrameProcessor:
    from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService

    return ElevenLabsRealtimeSTTService(
        api_key=_get_env("ELEVENLABS_API_KEY"),
        settings=ElevenLabsRealtimeSTTService.Settings(
            model="scribe_v2_realtime",
            language=Language.EN,
        ),
    )


def create_elevenlabs_http(aiohttp_session: "aiohttp.ClientSession") -> FrameProcessor:
    from pipecat.services.elevenlabs.stt import ElevenLabsSTTService

    return ElevenLabsSTTService(
        aiohttp_session=aiohttp_session,
        api_key=_get_env("ELEVENLABS_API_KEY"),
        settings=ElevenLabsSTTService.Settings(
            model="scribe_v2",
            language=Language.EN,
        ),
    )


def create_fal() -> FrameProcessor:
    from pipecat.services.fal.stt import FalSTTService

    return FalSTTService(
        api_key=_get_env("FAL_KEY"),
        settings=FalSTTService.Settings(
            language=Language.EN,
        ),
    )


def create_gladia() -> FrameProcessor:
    from pipecat.services.gladia.config import (
        LanguageConfig,
        PreProcessingConfig,
    )
    from pipecat.services.gladia.stt import GladiaSTTService

    return GladiaSTTService(
        api_key=_get_env("GLADIA_API_KEY"),
        region=os.getenv("GLADIA_REGION", "us-west"),
        settings=GladiaSTTService.Settings(
            model="solaria-1",
            language_config=LanguageConfig(
                languages=[Language.EN],
            ),
            endpointing=0.01,
            pre_processing=PreProcessingConfig(
                speech_threshold=0.8,
            ),
        ),
    )


def create_google() -> FrameProcessor:
    from pipecat.services.google.stt import GoogleSTTService

    return GoogleSTTService(
        credentials_path=_get_env("GOOGLE_APPLICATION_CREDENTIALS"),
        location=os.getenv("GOOGLE_LOCATION", "us-central1"),
        settings=GoogleSTTService.Settings(
            languages=Language.EN_US,
            model="latest_long",
        ),
    )


def create_gradium() -> FrameProcessor:
    from pipecat.services.gradium.stt import GradiumSTTService

    return GradiumSTTService(
        api_key=_get_env("GRADIUM_API_KEY"),
        settings=GradiumSTTService.Settings(
            model="default",
            language=Language.EN,
            delay_in_frames=12,
        ),
    )


def create_groq() -> FrameProcessor:
    from pipecat.services.groq.stt import GroqSTTService

    return GroqSTTService(
        api_key=_get_env("GROQ_API_KEY"),
        settings=GroqSTTService.Settings(
            model="whisper-large-v3-turbo",
            language=Language.EN,
        ),
    )


def create_mistral() -> FrameProcessor:
    from pipecat.services.mistral.stt import MistralSTTService

    return MistralSTTService(
        api_key=_get_env("MISTRAL_API_KEY"),
        settings=MistralSTTService.Settings(
            model="voxtral-mini-transcribe-realtime-2602",
            language=Language.EN,
        ),
    )


def create_nvidia() -> FrameProcessor:
    from pipecat.services.nvidia.stt import NvidiaSTTService

    return NvidiaSTTService(
        api_key=_get_env("NVIDIA_API_KEY"),
        settings=NvidiaSTTService.Settings(
            language=Language.EN_US,
        ),
    )


def create_nvidia_sagemaker() -> FrameProcessor:
    from pipecat.services.nvidia.sagemaker.stt import NvidiaSageMakerSTTService

    for env_var in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
    ):
        _set_env_from_config(env_var)

    return NvidiaSageMakerSTTService(
        endpoint_name=_get_env("SAGEMAKER_ASR_ENDPOINT_NAME"),
        region=_get_env_from_config("AWS_REGION") or "us-west-2",
    )


def create_openai() -> FrameProcessor:
    from pipecat.services.openai.stt import OpenAISTTService

    return OpenAISTTService(
        api_key=_get_env("OPENAI_API_KEY"),
        settings=OpenAISTTService.Settings(
            model="gpt-4o-mini-transcribe",
            language=Language.EN,
        ),
    )


def create_openai_realtime() -> FrameProcessor:
    from pipecat.services.openai.stt import OpenAIRealtimeSTTService

    return OpenAIRealtimeSTTService(
        api_key=_get_env("OPENAI_API_KEY"),
        settings=OpenAIRealtimeSTTService.Settings(
            model="gpt-4o-transcribe",
            language=Language.EN,
        ),
    )


def create_sarvam() -> FrameProcessor:
    from pipecat.services.sarvam.stt import SarvamSTTService

    return SarvamSTTService(
        api_key=_get_env("SARVAM_API_KEY"),
        settings=SarvamSTTService.Settings(
            model="saarika:v2.5",
        ),
    )


def create_smallest() -> FrameProcessor:
    from pipecat.services.smallest.stt import SmallestSTTService

    return SmallestSTTService(
        api_key=_get_env("SMALLEST_API_KEY"),
        settings=SmallestSTTService.Settings(
            language=Language.EN,
            model="pulse",
        ),
    )


def create_soniox() -> FrameProcessor:
    from pipecat.services.soniox.stt import SonioxSTTService

    return SonioxSTTService(
        api_key=_get_env("SONIOX_API_KEY"),
        settings=SonioxSTTService.Settings(
            model="stt-rt-v4",
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
        settings=SpeechmaticsSTTService.Settings(
            language=Language.EN,
            turn_detection_mode=TurnDetectionMode.EXTERNAL,
        ),
    )


def create_whisper() -> FrameProcessor:
    from pipecat.services.whisper.stt import Model, WhisperSTTService

    return WhisperSTTService(
        settings=WhisperSTTService.Settings(
            model=Model.DISTIL_MEDIUM_EN,
            language=Language.EN,
        ),
    )


def create_xai() -> FrameProcessor:
    from pipecat.services.xai.stt import XAISTTService

    return XAISTTService(
        api_key=_get_env("XAI_API_KEY"),
        settings=XAISTTService.Settings(
            language=Language.EN,
        ),
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
        vendor="AssemblyAI",
        model_label="universal-streaming-english",
        required_env_vars=["ASSEMBLYAI_API_KEY"],
        is_current=False,  # superseded by assemblyai_u3_rt_pro
    ),
    "assemblyai_u3_rt_pro": ServiceDefinition(
        factory=create_assemblyai_u3_rt_pro,
        vendor="AssemblyAI",
        model_label="u3-rt-pro",
        required_env_vars=["ASSEMBLYAI_API_KEY"],
    ),
    "aws": ServiceDefinition(
        factory=create_aws,
        vendor="AWS",
        model_label="N/A",
        required_env_vars=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
    ),
    "azure": ServiceDefinition(
        factory=create_azure,
        vendor="Azure",
        model_label="N/A",
        required_env_vars=["AZURE_SPEECH_API_KEY", "AZURE_SPEECH_REGION"],
    ),
    "cartesia": ServiceDefinition(
        factory=create_cartesia,
        vendor="Cartesia",
        model_label="ink-whisper",
        required_env_vars=["CARTESIA_API_KEY"],
        is_current=False,  # superseded by cartesia_ink2
    ),
    "cartesia_ink2": ServiceDefinition(
        factory=create_cartesia_ink2,
        vendor="Cartesia",
        model_label="ink-2",
        required_env_vars=["CARTESIA_API_KEY"],
    ),
    "deepgram": ServiceDefinition(
        factory=create_deepgram,
        vendor="Deepgram",
        model_label="nova-3-general",
        required_env_vars=["DEEPGRAM_API_KEY"],
    ),
    "elevenlabs": ServiceDefinition(
        factory=create_elevenlabs,
        vendor="ElevenLabs",
        model_label="scribe_v2_realtime",
        required_env_vars=["ELEVENLABS_API_KEY"],
    ),
    "elevenlabs_http": ServiceDefinition(
        factory=create_elevenlabs_http,
        vendor="ElevenLabs",
        model_label="scribe_v2",
        required_env_vars=["ELEVENLABS_API_KEY"],
        needs_aiohttp=True,
    ),
    "fal": ServiceDefinition(
        factory=create_fal,
        vendor="fal",
        model_label="N/A",
        required_env_vars=["FAL_KEY"],
    ),
    "gladia": ServiceDefinition(
        factory=create_gladia,
        vendor="Gladia",
        model_label="solaria-1",
        required_env_vars=["GLADIA_API_KEY"],
    ),
    "google": ServiceDefinition(
        factory=create_google,
        vendor="Google",
        model_label="latest-long",
        required_env_vars=["GOOGLE_APPLICATION_CREDENTIALS"],
    ),
    "gradium": ServiceDefinition(
        factory=create_gradium,
        vendor="Gradium",
        model_label="default",
        required_env_vars=["GRADIUM_API_KEY"],
    ),
    "groq": ServiceDefinition(
        factory=create_groq,
        vendor="Groq",
        model_label="whisper-large-v3-turbo",
        required_env_vars=["GROQ_API_KEY"],
    ),
    "mistral": ServiceDefinition(
        factory=create_mistral,
        vendor="Mistral",
        model_label="voxtral-mini-transcribe-realtime-2602",
        required_env_vars=["MISTRAL_API_KEY"],
    ),
    "nvidia": ServiceDefinition(
        factory=create_nvidia,
        vendor="NVIDIA",
        model_label="nemotron-asr-streaming",
        required_env_vars=["NVIDIA_API_KEY"],
    ),
    "nvidia_sagemaker": ServiceDefinition(
        factory=create_nvidia_sagemaker,
        vendor="NVIDIA",
        model_label="cache-aware-parakeet-rnnt-en-US-asr-streaming-sortformer",
        required_env_vars=["SAGEMAKER_ASR_ENDPOINT_NAME"],
    ),
    "openai": ServiceDefinition(
        factory=create_openai,
        vendor="OpenAI",
        model_label="gpt-4o-mini-transcribe",
        required_env_vars=["OPENAI_API_KEY"],
    ),
    "openai_realtime": ServiceDefinition(
        factory=create_openai_realtime,
        vendor="OpenAI",
        model_label="gpt-4o-transcribe",
        required_env_vars=["OPENAI_API_KEY"],
    ),
    "sarvam": ServiceDefinition(
        factory=create_sarvam,
        vendor="Sarvam",
        model_label="saarika:v2.5",
        required_env_vars=["SARVAM_API_KEY"],
    ),
    "smallest": ServiceDefinition(
        factory=create_smallest,
        vendor="Smallest AI",
        model_label="pulse",
        required_env_vars=["SMALLEST_API_KEY"],
    ),
    "soniox": ServiceDefinition(
        factory=create_soniox,
        vendor="Soniox",
        model_label="stt-rt-v4",
        required_env_vars=["SONIOX_API_KEY"],
    ),
    "speechmatics": ServiceDefinition(
        factory=create_speechmatics,
        vendor="Speechmatics",
        model_label="N/A",
        required_env_vars=["SPEECHMATICS_API_KEY"],
    ),
    "whisper": ServiceDefinition(
        factory=create_whisper,
        vendor="Whisper",
        model_label="Distil-Whisper Medium EN (local)",
        required_env_vars=[],  # Local model, no API key needed
    ),
    "xai": ServiceDefinition(
        factory=create_xai,
        vendor="xAI",
        model_label="N/A",
        required_env_vars=["XAI_API_KEY"],
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


def get_display_names(service_names: list) -> dict[str, str]:
    """Build human-readable labels for plots/reports from registry metadata.

    Each label is the vendor name, disambiguated with the model label only when
    the same vendor appears more than once in ``service_names`` — so a lone
    vendor reads "AssemblyAI" while two Cartesia models read "Cartesia Ink-Whisper"
    and "Cartesia Ink-2". Accepts service keys or ServiceName values; unknown
    names are skipped.

    This derives labels from ``vendor`` / ``model_label`` so callers don't keep a
    separate hand-maintained name map in sync with the registry.
    """
    keys = [getattr(n, "value", n) for n in service_names]
    vendor_counts = Counter(STT_SERVICES[k].vendor for k in keys if k in STT_SERVICES)

    labels: dict[str, str] = {}
    for key in keys:
        definition = STT_SERVICES.get(key)
        if definition is None:
            continue
        if vendor_counts[definition.vendor] > 1:
            labels[key] = f"{definition.vendor} {definition.model_label}"
        else:
            labels[key] = definition.vendor
    return labels


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


def _set_env_from_config(env_var_name: str) -> None:
    """Set os.environ from config when a value exists and the env var is unset."""
    if os.getenv(env_var_name):
        return

    value = _get_env_from_config(env_var_name)
    if value:
        os.environ[env_var_name] = value


def is_service_available(name: str) -> bool:
    """Check if a service has all required environment variables set."""
    if name not in STT_SERVICES:
        return False
    definition = STT_SERVICES[name]
    return all(_get_env_from_config(env_var) for env_var in definition.required_env_vars)


def create_stt_service(
    service_name: "ServiceName",
    aiohttp_session: "aiohttp.ClientSession | None" = None,
) -> FrameProcessor:
    """Create an STT service instance using its factory function.

    Args:
        service_name: The STT service to create.
        aiohttp_session: Optional aiohttp session for services that require one
            (i.e. services with needs_aiohttp=True in their ServiceDefinition).

    Returns:
        Configured STT service instance.

    Raises:
        ValueError: If service_name is not supported or required credentials are missing.
    """
    from loguru import logger

    definition = get_service_definition(service_name.value)
    logger.debug(f"Creating {service_name.value} STT service")

    if definition.needs_aiohttp:
        if aiohttp_session is None:
            raise ValueError(
                f"Service {service_name.value} requires an aiohttp session "
                f"but none was provided. The pipeline runner should create one."
            )
        return definition.factory(aiohttp_session)

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
