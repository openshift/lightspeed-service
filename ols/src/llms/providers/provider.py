"""LLM provider class definition."""

import abc
import logging
from dataclasses import dataclass
from typing import Any, Optional

from ibm_watson_machine_learning.metanames import (
    GenTextParamsMetaNames as GenParams,
)
from langchain.llms.base import LLM

from ols import config
from ols.app.models.config import ProviderConfig
from ols.constants import (
    PROVIDER_AZURE_OPENAI,
    PROVIDER_BAM,
    PROVIDER_OPENAI,
    PROVIDER_WATSONX,
    GenericLLMParameters,
)

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
    ProviderParameter(GenParams.DECODING_METHOD, str),
    ProviderParameter(GenParams.MIN_NEW_TOKENS, int),
    ProviderParameter(GenParams.MAX_NEW_TOKENS, int),
    ProviderParameter(GenParams.RANDOM_SEED, int),
    ProviderParameter(GenParams.TOP_K, int),
    ProviderParameter(GenParams.TOP_P, float),
    ProviderParameter(GenParams.TEMPERATURE, float),
    ProviderParameter(GenParams.REPETITION_PENALTY, float),
}

# available parameters for all supported LLM providers
available_provider_parameters: dict[str, set[ProviderParameter]] = {
    PROVIDER_AZURE_OPENAI: AzureOpenAIParameters,
    PROVIDER_OPENAI: OpenAIParameters,
    PROVIDER_BAM: BAMParameters,
    PROVIDER_WATSONX: WatsonxParameters,
}

# Generic to Azure OpenAI parameters mapping
AzureOpenAIParametersMapping: dict[str, str] = {}

# Generic to OpenAI parameters mapping
OpenAIParametersMapping: dict[str, str] = {}

# Generic to BAM parameters mapping
BAMParametersMapping: dict[str, str] = {}

# Generic to Watsonx parameters mapping
WatsonxParametersMapping: dict[str, str] = {
    GenericLLMParameters.MIN_NEW_TOKENS: GenParams.MIN_NEW_TOKENS,
    GenericLLMParameters.MAX_NEW_TOKENS: GenParams.MAX_NEW_TOKENS,
    GenericLLMParameters.TOP_K: GenParams.TOP_K,
    GenericLLMParameters.TOP_P: GenParams.TOP_P,
    GenericLLMParameters.TEMPERATURE: GenParams.TEMPERATURE,
}

# mapping between generic parameters and LLM-specific parameters
generic_to_llm_parameters: dict[str, dict[str, str]] = {
    PROVIDER_AZURE_OPENAI: AzureOpenAIParametersMapping,
    PROVIDER_OPENAI: OpenAIParametersMapping,
    PROVIDER_BAM: BAMParametersMapping,
    PROVIDER_WATSONX: WatsonxParametersMapping,
}


class AbstractLLMProvider(abc.ABC):
    """Abstract class defining `LLMProvider` interface."""

    @property
    @abc.abstractmethod
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
            params: The optional parameters that will be converted into LLM specific ones.
        """
        self.model = model
        self.provider_config = provider_config
        params = self._override_params(params or {})
        params = self._remap_to_llm_params(params)
        self.params = self._validate_parameters(params)

    def _remap_to_llm_params(
        self, generic_llm_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Remap generic parameters into LLM specific ones."""
        if self.provider_config is None:
            logger.warning("Provider is not set. Parameters mapping won't proceed.")
            return generic_llm_params

        provider = self.provider_config.type

        if provider is None:
            logger.warning(
                "Provider type is not set. Parameters mapping won't proceed."
            )
            return generic_llm_params

        if provider not in generic_to_llm_parameters:
            logger.warning(f"Mappings for provider {provider} are not defined.")
            return generic_llm_params

        # retrieve mapping
        mapping = generic_to_llm_parameters[provider]

        llm_parameters = {}

        # map parameters
        for key, value in generic_llm_params.items():
            if key in mapping:
                new_key = mapping[key]
                llm_parameters[new_key] = value
            else:
                llm_parameters[key] = value

        return llm_parameters

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

        # config params overrides everything
        if config.dev_config.llm_params:
            logger.debug(
                f"overriding LLM params with debug options {config.dev_config.llm_params}"
            )
            updated_params = {**updated_params, **config.dev_config.llm_params}

        return updated_params
