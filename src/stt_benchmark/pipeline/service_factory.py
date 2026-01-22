"""Factory for creating Pipecat STT services."""

import importlib

import aiohttp
from loguru import logger
from pipecat.processors.frame_processor import FrameProcessor

from stt_benchmark.config import get_config
from stt_benchmark.models import ServiceName
from stt_benchmark.services import STT_SERVICES, get_service_config


def _get_env_value(env_var_name: str) -> str:
    """Get environment variable value from config (supports .env files via Pydantic).

    Derives config attribute from env var name: DEEPGRAM_API_KEY -> deepgram_api_key
    """
    config = get_config()
    attr_name = env_var_name.lower()
    return getattr(config, attr_name, "")


async def create_stt_service(
    service_name: ServiceName,
    aiohttp_session: aiohttp.ClientSession | None = None,
    model: str | None = None,
) -> FrameProcessor:
    """Create an STT service instance from configuration.

    Args:
        service_name: The STT service to create.
        aiohttp_session: Optional aiohttp session for services that need it.
        model: Optional model name override.

    Returns:
        Configured STT service instance.

    Raises:
        ValueError: If service_name is not supported or required API key is missing.
    """
    # Get service configuration
    config = get_service_config(service_name.value)

    # Build kwargs from environment variables
    kwargs = {}

    # Handle simple api_key_env case
    if config.api_key_env:
        api_key = _get_env_value(config.api_key_env)
        if not api_key:
            raise ValueError(f"{config.api_key_env} environment variable not set")
        kwargs["api_key"] = api_key

    # Handle env_vars dict (for services with multiple credentials)
    for param_name, env_var_name in config.env_vars.items():
        value = _get_env_value(env_var_name)
        if not value:
            raise ValueError(f"{env_var_name} environment variable not set")
        kwargs[param_name] = value

    # Add extra static kwargs
    kwargs.update(config.extra_kwargs)

    # Check aiohttp requirement
    if config.needs_aiohttp and aiohttp_session is None:
        raise ValueError(f"{config.name} STT requires an aiohttp session")

    # Dynamically import the service class
    module = importlib.import_module(config.pipecat_module)
    service_class = getattr(module, config.pipecat_class)

    # Add model if supported
    if config.default_model is not None or model is not None:
        kwargs["model"] = model or config.default_model

    # Note: We don't pass language - services default to English
    # and each service handles the language parameter differently

    # Add aiohttp session if needed
    if config.needs_aiohttp:
        kwargs["aiohttp_session"] = aiohttp_session

    logger.debug(f"Creating {config.name} STT service with model={kwargs.get('model')}")
    return service_class(**kwargs)


def get_available_services() -> list[ServiceName]:
    """Get list of services that have all required credentials configured.

    Returns:
        List of ServiceName values for available services.
    """
    available = []

    for name, config in STT_SERVICES.items():
        # Check api_key_env if set
        if config.api_key_env:
            api_key = _get_env_value(config.api_key_env)
            if not api_key:
                logger.debug(f"Service {name} not available (no {config.api_key_env})")
                continue

        # Check all env_vars are set
        all_env_vars_set = True
        for _, env_var_name in config.env_vars.items():
            value = _get_env_value(env_var_name)
            if not value:
                logger.debug(f"Service {name} not available (no {env_var_name})")
                all_env_vars_set = False
                break

        if not all_env_vars_set:
            continue

        # Service has all required credentials
        try:
            available.append(ServiceName(name))
        except ValueError:
            logger.warning(f"Service {name} not in ServiceName enum")

    return available


def get_all_services() -> list[ServiceName]:
    """Get list of all supported services.

    Returns:
        List of all ServiceName values.
    """
    return list(ServiceName)


def service_needs_aiohttp(service_name: ServiceName) -> bool:
    """Check if a service requires an aiohttp session.

    Args:
        service_name: The service to check.

    Returns:
        True if the service needs an aiohttp session.
    """
    config = get_service_config(service_name.value)
    return config.needs_aiohttp
