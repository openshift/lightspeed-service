"""LLM backend libraries loader."""

import logging
from typing import Any, Optional

from langchain.llms.base import LLM

from ols import config, constants
from ols.app.models.config import LLMProviders, ProviderConfig
from ols.src.llms.providers.registry import LLMProvidersRegistry

logger = logging.getLogger(__name__)


class LLMConfigurationError(Exception):
    """LLM configuration is wrong."""


class UnknownProviderError(LLMConfigurationError):
    """No configuration for provider."""


class UnsupportedProviderError(LLMConfigurationError):
    """Provider is not supported."""


class ModelConfigMissingError(LLMConfigurationError):
    """No configuration exists for the requested model name."""


def resolve_provider_config(
    provider: str, model: str, providers_config: LLMProviders
) -> ProviderConfig:
    """Ensure the provided inputs (provider/model) are valid in config.

    Return respective provider configuration.
    """
    if provider not in providers_config.providers:
        raise UnknownProviderError(
            f"Provider '{provider}' is not a valid provider. "
            f"Valid providers are: {list(providers_config.providers.keys())}"
        )

    provider_config = providers_config.providers.get(provider)

    if model not in provider_config.models:
        raise ModelConfigMissingError(
            f"Model '{model}' is not a valid model for provider '{provider}'. "
            f"Valid models are: {list(provider_config.models.keys())}"
        )

    return provider_config


def load_llm(
    provider: str,
    model: str,
    generic_llm_params: Optional[dict] = None,
) -> LLM | Any:  # Temporarily using Any, as mypy gives error for missing bind_tools
    """Load LLM according to input provider and model.

    Args:
        provider: The provider name.
        model: The model name.
        generic_llm_params: The optional parameters that will be converted into LLM-specific ones.

    Raises:
        LLMConfigurationError: If the whole provider configuration is missing.
        UnsupportedProviderError: If the provider is not supported (implemented).
        UnknownProviderError: If the provider is not known.
        ModelConfigMissingError: If the model configuration is missing.

    Example:
        ```python
        # using the class and overriding specific parameters
        generic_llm_params = {'temperature': 0.02, 'top_p': 0.95}

        bare_llm = load_llm(provider="provider-name", model="model-name",
                            generic_llm_params=generic_llm_params).llm
        llm_chain = prompt | bare_llm
        ```
    """
    providers_config = config.config.llm_providers
    if providers_config is None:
        raise LLMConfigurationError(
            f"Providers configuration missing in {constants.DEFAULT_CONFIGURATION_FILE}"
        )
    llm_providers_reg = LLMProvidersRegistry

    provider_config = resolve_provider_config(provider, model, providers_config)
    if provider_config.type not in llm_providers_reg.llm_providers:
        raise UnsupportedProviderError(
            f"Unsupported LLM provider type '{provider_config.type}'."
        )

    logger.debug("loading LLM model '%s' from provider '%s'", model, provider)

    llm_provider = llm_providers_reg.llm_providers[provider_config.type]
    return llm_provider(model, provider_config, generic_llm_params or {}).load()
