"""LLM provider class definition."""

import abc
import logging
import ssl
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain.llms.base import LLM
from langchain_core.language_models.chat_models import BaseChatModel

from ols import config
from ols.app.models.config import ProviderConfig
from ols.constants import (
    PROVIDER_AZURE_OPENAI,
    PROVIDER_BAM,
    PROVIDER_FAKE,
    PROVIDER_OPENAI,
    PROVIDER_RHELAI_VLLM,
    PROVIDER_RHOAI_VLLM,
    PROVIDER_WATSONX,
    GenericLLMParameters,
)
from ols.utils import tls

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderParameter:
    """Specification of one provider parameter."""

    name: str
    _type: type


AzureOpenAIParameters = {
    ProviderParameter("azure_endpoint", str),
    ProviderParameter("api_key", str),
    ProviderParameter("api_version", str),
    ProviderParameter("azure_ad_token", str),
    ProviderParameter("base_url", str),
    ProviderParameter("deployment_name", str),
    ProviderParameter("model", str),
    ProviderParameter("model_kwargs", dict),
    ProviderParameter("organization", str),
    ProviderParameter("cache", str),
    ProviderParameter("temperature", float),
    ProviderParameter("max_tokens", int),
    ProviderParameter("verbose", bool),
    ProviderParameter("http_client", httpx.Client),
    ProviderParameter("http_async_client", httpx.AsyncClient),
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
    ProviderParameter("temperature", float),
    ProviderParameter("max_tokens", int),
    ProviderParameter("verbose", bool),
    ProviderParameter("http_client", httpx.Client),
    ProviderParameter("http_async_client", httpx.AsyncClient),
}

RHOAIVLLMParameters = {
    ProviderParameter("openai_api_key", str),
    ProviderParameter("api_version", str),
    ProviderParameter("base_url", str),
    ProviderParameter("deployment_name", str),
    ProviderParameter("model", str),
    ProviderParameter("model_kwargs", dict),
    ProviderParameter("organization", str),
    ProviderParameter("cache", str),
    ProviderParameter("temperature", float),
    ProviderParameter("max_tokens", int),
    ProviderParameter("verbose", bool),
    ProviderParameter("http_client", httpx.Client),
    ProviderParameter("http_async_client", httpx.AsyncClient),
}

RHELAIVLLMParameters = {
    ProviderParameter("openai_api_key", str),
    ProviderParameter("api_version", str),
    ProviderParameter("base_url", str),
    ProviderParameter("deployment_name", str),
    ProviderParameter("model", str),
    ProviderParameter("model_kwargs", dict),
    ProviderParameter("organization", str),
    ProviderParameter("cache", str),
    ProviderParameter("temperature", float),
    ProviderParameter("max_tokens", int),
    ProviderParameter("verbose", bool),
    ProviderParameter("http_client", httpx.Client),
    ProviderParameter("http_async_client", httpx.AsyncClient),
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

FakeProviderParameters = {
    ProviderParameter("stream", bool),
    ProviderParameter("response", str),
    ProviderParameter("chunks", int),
    ProviderParameter("sleep", float),
}

# available parameters for all supported LLM providers
available_provider_parameters: dict[str, set[ProviderParameter]] = {
    PROVIDER_AZURE_OPENAI: AzureOpenAIParameters,
    PROVIDER_OPENAI: OpenAIParameters,
    PROVIDER_RHELAI_VLLM: RHELAIVLLMParameters,
    PROVIDER_RHOAI_VLLM: RHOAIVLLMParameters,
    PROVIDER_BAM: BAMParameters,
    PROVIDER_WATSONX: WatsonxParameters,
    PROVIDER_FAKE: FakeProviderParameters,
}

# Generic to Azure OpenAI parameters mapping
AzureOpenAIParametersMapping: dict[str, str] = {
    GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: "max_tokens",
}

# Generic to OpenAI parameters mapping
OpenAIParametersMapping: dict[str, str] = {
    GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: "max_tokens",
}

# Generic to RHELAI VLLM parameters mapping
RHELAIVLLMParametersMapping: dict[str, str] = {
    GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: "max_tokens",
}

# Generic to RHOAI VLLM parameters mapping
RHOAIVLLMParametersMapping: dict[str, str] = {
    GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: "max_tokens",
}

# Generic to BAM parameters mapping
BAMParametersMapping: dict[str, str] = {
    GenericLLMParameters.MIN_TOKENS_FOR_RESPONSE: "min_new_tokens",
    GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: "max_new_tokens",
}

# Generic to Watsonx parameters mapping
WatsonxParametersMapping: dict[str, str] = {
    GenericLLMParameters.MIN_TOKENS_FOR_RESPONSE: GenParams.MIN_NEW_TOKENS,
    GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: GenParams.MAX_NEW_TOKENS,
    GenericLLMParameters.TOP_K: GenParams.TOP_K,
    GenericLLMParameters.TOP_P: GenParams.TOP_P,
    GenericLLMParameters.TEMPERATURE: GenParams.TEMPERATURE,
}

# Generic to fake parameter mapping
FakeProviderParametersMapping: dict[str, str] = {}

# mapping between generic parameters and LLM-specific parameters
generic_to_llm_parameters: dict[str, dict[str, str]] = {
    PROVIDER_AZURE_OPENAI: AzureOpenAIParametersMapping,
    PROVIDER_OPENAI: OpenAIParametersMapping,
    PROVIDER_RHELAI_VLLM: RHELAIVLLMParametersMapping,
    PROVIDER_RHOAI_VLLM: RHOAIVLLMParametersMapping,
    PROVIDER_BAM: BAMParametersMapping,
    PROVIDER_WATSONX: WatsonxParametersMapping,
    PROVIDER_FAKE: FakeProviderParametersMapping,
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
    def load(self) -> BaseChatModel | LLM:
        """Load and langchain `LLM` instance and return it."""


class LLMProvider(AbstractLLMProvider):
    """LLM provider base class."""

    def __init__(
        self,
        model: str,
        provider_config: ProviderConfig,
        params: Optional[dict] = None,
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
            logger.warning("Mappings for provider %s are not defined.", provider)
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
                "Available parameters for provider %s are not defined."
                "Parameters validation is disabled.",
                provider,
            )
            return params

        # retrieve all available parameters for provider
        available_parameters = available_provider_parameters[provider]
        available_parameter_names = {p.name for p in available_parameters}

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
                    filtered_params[parameter_name] = None
                    continue
                # other parameters
                logger.warning(
                    "Parameter %s with type %s can not be used by provider %s",
                    parameter_name,
                    parameter_type,
                    provider,
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
                "overriding LLM params with debug options %s",
                config.dev_config.llm_params,
            )
            updated_params = {**updated_params, **config.dev_config.llm_params}

        return updated_params

    def _construct_httpx_client(
        self, use_custom_certificate_store: bool, use_async: bool
    ) -> httpx.Client | httpx.AsyncClient:
        """Construct HTTPX client instance to be used to communicate with LLM."""
        # no proxy by default
        proxy = None
        # set up proxy if configured.
        if config.ols_config.proxy_config and config.ols_config.proxy_config.proxy_url:
            logger.debug(
                "Proxy is configured. Proxy URL: %s Proxy CA cert: %s",
                config.ols_config.proxy_config.proxy_url,
                config.ols_config.proxy_config.proxy_ca_cert_path,
            )
            proxy_context = None
            if config.ols_config.proxy_config.is_https():
                proxy_context = ssl.create_default_context(
                    cafile=config.ols_config.proxy_config.proxy_ca_cert_path
                )
            proxy = httpx.Proxy(
                url=config.ols_config.proxy_config.proxy_url, ssl_context=proxy_context
            )

        sec_profile = self.provider_config.tls_security_profile

        # if security profile is not set, use httpx client as is
        if sec_profile is None or sec_profile.profile_type is None:
            verify: ssl.SSLContext | bool = True
            if use_custom_certificate_store:
                logger.debug(
                    "Custom Certificate store location: %s",
                    self.provider_config.certificates_store,
                )
                custom_context = ssl.create_default_context()
                custom_context.check_hostname = False
                custom_context.load_verify_locations(
                    cafile=self.provider_config.certificates_store
                )
                verify = custom_context
            if use_async:
                return httpx.AsyncClient(verify=verify, proxies=proxy)
            return httpx.Client(verify=verify, proxies=proxy)

        # security profile is set -> we need to retrieve SSL version and list of allowed ciphers
        ciphers = tls.ciphers_as_string(sec_profile.ciphers, sec_profile.profile_type)
        logger.info("list of ciphers: %s", ciphers)

        min_tls_version = tls.min_tls_version(
            sec_profile.min_tls_version, sec_profile.profile_type
        )
        logger.info("min TLS version: %s", min_tls_version)

        ssl_version = tls.ssl_tls_version(min_tls_version)
        logger.info("SSL version: %d", ssl_version)

        context = ssl.create_default_context()

        if ssl_version is not None:
            context.minimum_version = ssl_version

        if ciphers is not None:
            context.set_ciphers(ciphers)

        if use_custom_certificate_store:
            context.load_verify_locations(self.provider_config.certificates_store)
        if use_async:
            return httpx.AsyncClient(verify=context, proxies=proxy)
        return httpx.Client(verify=context, proxies=proxy)
