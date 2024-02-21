"""LLM backend libraries loader."""

import logging
from typing import Optional

from langchain.llms.base import LLM

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.registry import LLMProvidersRegistry
from ols.utils import config

logger = logging.getLogger(__name__)


class LLMConfigurationError(Exception):
    """LLM configuration is wrong."""


class UnknownProviderError(LLMConfigurationError):
    """No configuration for provider."""


class UnsupportedProviderError(LLMConfigurationError):
    """Provider is not supported."""


class ModelConfigMissingError(LLMConfigurationError):
    """No configuration exists for the requested model name."""


def _resolve_provider_config(provider, model, providers_config) -> ProviderConfig:
    """Ensure the provided inputs (provider/model) are valid in config.

    Return respective provider configuration.
    """
    if provider not in providers_config.providers:
        raise UnknownProviderError(
            f"Provider '{provider}' is not a valid provider! "
            f"Valid providers are: {list(providers_config.providers.keys())}"
        )

    provider_config = providers_config.providers.get(provider)

    if model not in provider_config.models:
        raise ModelConfigMissingError(
            f"Model '{model}' is not a valid model for provider '{provider}'! "
            f"Valid models are: {list(provider_config.models.keys())}"
        )

    return provider_config


def load_llm(provider: str, model: str, llm_params: Optional[dict] = None) -> LLM:
    """Load LLM according to input provider and model.

    Args:
        provider: The provider name.
        model: The model name.
        llm_params: The optional LLM parameters.

    Raises:
        UnsupportedProviderError: If the provider is not supported (implemented).
        UnknownProviderError: If the provider is not known.
        ModelConfigMissingError: If the model configuration is missing.

    Example:
        ```python
        # using the class and overriding specific parameters
        llm_params = {'temperature': 0.02, 'top_p': 0.95}

        bare_llm = load_llm(provider="openai", model="gpt-3.5-turbo", llm_params=llm_params).llm
        llm_chain = LLMChain(llm=bare_llm, prompt=prompt)
        ```
    """
    providers_config = config.config.llm_providers
    llm_providers_reg = LLMProvidersRegistry

    provider_config = _resolve_provider_config(provider, model, providers_config)
    if provider_config.type not in llm_providers_reg.llm_providers:
        raise UnsupportedProviderError(
            f"Unsupported LLM provider type '{provider_config.type}'!"
        )

    logger.debug(f"loading LLM '{model}' from '{provider}'")

    llm_provider = llm_providers_reg.llm_providers[provider_config.type]
    llm = llm_provider(model, provider_config, llm_params or {}).load()

    return llm
