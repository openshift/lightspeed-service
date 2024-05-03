"""LLM provider class definition."""

import abc
import logging
from dataclasses import dataclass
from typing import Any, Optional

from langchain.llms.base import LLM

from ols.app.models.config import ProviderConfig
from ols.constants import (
    PROVIDER_AZURE_OPENAI,
    PROVIDER_BAM,
    PROVIDER_OPENAI,
    PROVIDER_WATSONX,
)
from ols.utils.config import ConfigManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderParameter:
    """Specification of one provider parameter."""

    _name: str
    _type: type


AzureOpenAIParameters = {
    ProviderParameter("azure_endpoint", str),
    ProviderParameter("api_key", str),
    ProviderParameter("api_version", str),
    ProviderParameter("base_url", str),
    ProviderParameter("deployment_name", str),
    ProviderParameter("model", str),
    ProviderParameter("model_kwargs", dict),
    ProviderParameter("organization", str),
    ProviderParameter("cache", str),
    ProviderParameter("streaming", bool),
    ProviderParameter("temperature", float),
    ProviderParameter("max_tokens", int),
    ProviderParameter("verbose", bool),
}

OpenAIParameters = {
    ProviderParameter("azure_endpoint", str),
    ProviderParameter("openai_api_key", str),
    ProviderParameter("api_version", str),
    ProviderParameter("base_url", str),
    ProviderParameter("deployment_name", str),
    ProviderParameter("model", str),
    ProviderParameter("model_kwargs", dict),
    ProviderParameter("organization", str),
    ProviderParameter("cache", str),
    ProviderParameter("streaming", bool),
    ProviderParameter("temperature", float),
    ProviderParameter("max_tokens", int),
    ProviderParameter("verbose", bool),
}

BAMParameters = {
    ProviderParameter("decoding_method", str),
    ProviderParameter("max_new_tokens", int),
    ProviderParameter("min_new_tokens", int),
    ProviderParameter("random_seed", int),
    ProviderParameter("top_k", int),
    ProviderParameter("top_p", float),
    ProviderParameter("repetition_penalty", float),
    ProviderParameter("temperature", float),
}

WatsonxParameters = {
    ProviderParameter("decoding_method", str),
    ProviderParameter("max_new_tokens", int),
    ProviderParameter("min_new_tokens", int),
    ProviderParameter("random_seed", int),
    ProviderParameter("top_k", int),
    ProviderParameter("top_p", float),
    ProviderParameter("repetition_penalty", float),
    ProviderParameter("temperature", float),
}

# available parameters for all supported LLM providers
available_provider_parameters: dict[str, set[ProviderParameter]] = {
    PROVIDER_AZURE_OPENAI: AzureOpenAIParameters,
    PROVIDER_OPENAI: OpenAIParameters,
    PROVIDER_BAM: BAMParameters,
    PROVIDER_WATSONX: WatsonxParameters,
}


class AbstractLLMProvider(abc.ABC):
    """Abstract class defining `LLMProvider` interface."""

    @abc.abstractproperty
    def default_params(self) -> dict:
        """Defaults LLM params.

        These will be overriden by the input parameters of the caller or
        via developer config.
        """

    @abc.abstractmethod
    def load(self) -> LLM:
        """Load and langchain `LLM` instance and return it."""


class LLMProvider(AbstractLLMProvider):
    """LLM provider base class."""

    def __init__(
        self, model: str, provider_config: ProviderConfig, params: Optional[dict] = None
    ) -> None:
        """Initialize LLM provider.

        Args:
            model: The model name.
            provider_config: The provider configuration.
            params: The optional LLM parameters.
        """
        self.model = model
        self.provider_config = provider_config
        params = self._override_params(params or {})
        self.params = self._validate_parameters(params)

    def _validate_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate parameters for LLM provider."""
        if self.provider_config is None:
            logger.warning("Provider is not set. Parameters validation is disabled.")
            return params

        provider = self.provider_config.type

        if provider is None:
            logger.warning(
                "Provider type is not set. Parameters validation is disabled."
            )
            return params

        if provider not in available_provider_parameters:
            logger.warning(
                f"Available parameters for provider {provider} are not defined."
                "Parameters validation is disabled."
            )
            return params

        # retrieve all available parameters for provider
        available_parameters = available_provider_parameters[provider]
        available_parameter_names = {p._name for p in available_parameters}

        # only supported parameters will be returned
        filtered_params: dict[str, Any] = {}

        for parameter_name, parameter_value in params.items():
            parameter_type = type(parameter_value)
            parameter = ProviderParameter(parameter_name, parameter_type)
            if parameter not in available_parameters:
                # check for allowed parameter with None value
                if (
                    parameter_value is None
                    and parameter_name in available_parameter_names
                ):
                    [p._name for p in AzureOpenAIParameters]
                    filtered_params[parameter_name] = None
                    continue
                # other parameters
                logger.warning(
                    f"Parameter {parameter_name} with type {parameter_type} "
                    f"can not be used by provider {provider}"
                )
            else:
                filtered_params[parameter_name] = parameter_value
        return filtered_params

    def _override_params(self, params: dict[Any, Any]) -> dict[Any, Any]:
        """Override LLM parameters if defined in developer config."""
        # input params overrides default params
        updated_params = {**self.default_params, **params}
        config_manager = ConfigManager()
        llm_params = config_manager.get_dev_config().llm_params

        # config params overrides everything
        if llm_params:
            logger.debug(f"overriding LLM params with debug options {llm_params}")
            updated_params = {
                **updated_params,
                **llm_params,
            }

        return updated_params
