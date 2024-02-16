"""LLM backend libraries loader."""

import inspect
import logging
import warnings
from typing import Optional

from genai import Client, Credentials
from genai.extensions.langchain import LangChainInterface
from genai.text.generation import TextGenerationParameters
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import (
    WatsonxLLM,
)
from ibm_watson_machine_learning.metanames import (
    GenTextParamsMetaNames as GenParams,
)
from langchain.llms.base import LLM
from langchain_openai import ChatOpenAI

from ols import constants
from ols.app.models.config import ProviderConfig
from ols.utils import config

logger = logging.getLogger(__name__)


# workaround to disable UserWarning
warnings.simplefilter("ignore", UserWarning)


class LLMConfigurationError(Exception):
    """LLM configuration is wrong."""


class UnknownProviderError(LLMConfigurationError):
    """No configuration for provider."""


class UnsupportedProviderError(LLMConfigurationError):
    """Provider is not supported."""


class ModelConfigMissingError(LLMConfigurationError):
    """No configuration exists for the requested model name."""


class ModelConfigInvalidError(LLMConfigurationError):
    """Model configuration is not valid."""


class LLMLoader:
    """Load LLM backend.

    Example:
        ```python
        # using the class and overriding specific parameters
        llm_params = {'temperature': 0.02, 'top_p': 0.95}

        bare_llm = LLMLoader(provider="openai", model="gpt-3.5-turbo", llm_params=llm_params).llm
        llm_chain = LLMChain(llm=bare_llm, prompt=prompt)
        ```
    """

    def __init__(
        self,
        provider: str,
        model: str,
        llm_params: Optional[dict] = None,
    ) -> None:
        """Initialize LLM loader.

        Args:
            provider: The LLM provider name.
            model: The LLM model name.
            llm_params: The parameters to override and pass to the LLM backend.

        Raises:
            UnknownProviderError: When no configuration exists for the requested provider.
            UnsupportedProviderError: When provider is unsupported.
            ModelConfigMissingError: When no configuration exists for the requested model name.
        """
        self.provider = provider
        self.model = model
        self.llm_params = llm_params or {}
        self.llm_providers = config.config.llm_providers
        self.provider_config = self._resolve_provider_config()

        self.llm = self._llm_instance()

    def _resolve_provider_config(self) -> ProviderConfig:
        """Ensure the provided inputs (provider/model) are valid in config."""
        if self.provider not in self.llm_providers.providers:
            raise UnknownProviderError(
                f"Provider '{self.provider}' is not a valid provider. "
                f"Valid providers are: {list(self.llm_providers.providers.keys())}"
            )

        provider_conf = self.llm_providers.providers.get(self.provider)

        if self.model not in provider_conf.models:
            raise ModelConfigMissingError(
                f"Model '{self.model}' is not a valid model for provider '{self.provider}'. "
                f"Valid models are: {list(provider_conf.models.keys())}"
            )

        return provider_conf

    def _get_llm_url(self, default: str) -> str:
        return (
            self.provider_config.models[self.model].url
            if self.provider_config.models[self.model].url is not None
            else (
                self.provider_config.url
                if self.provider_config.url is not None
                else default
            )
        )

    def _get_llm_credentials(self) -> str:
        return (
            self.provider_config.models[self.model].credentials
            if self.provider_config.models[self.model].credentials is not None
            else self.provider_config.credentials
        )

    def _llm_instance(self) -> LLM:
        logger.debug(
            f"[{inspect.stack()[0][3]}] Loading LLM {self.model} from {self.provider}"
        )
        # convert to string to handle None or False definitions
        match str(self.provider_config.type):
            case constants.PROVIDER_OPENAI:
                return self._openai_llm_instance(
                    self._get_llm_url("https://api.openai.com/v1"),
                    self._get_llm_credentials(),
                )
            # case constants.PROVIDER_OLLAMA:
            #     return self._ollama_llm_instance()
            # case constants.PROVIDER_TGI:
            #     return self._tgi_llm_instance()
            case constants.PROVIDER_WATSONX:
                return self._watson_llm_instance(
                    self._get_llm_url("https://us-south.ml.cloud.ibm.com"),
                    self._get_llm_credentials(),
                    self.provider_config.project_id,
                )
            case constants.PROVIDER_BAM:
                return self._bam_llm_instance(
                    self._get_llm_url("https://bam-api.res.ibm.com"),
                    self._get_llm_credentials(),
                )
            case _:
                msg = (
                    f"Unsupported LLM provider type {self.provider_config.type} "
                    f"in provider {self.provider}"
                )
                logger.error(msg)
                raise UnsupportedProviderError(msg)

    def _openai_llm_instance(self, api_url: str, api_key: str) -> LLM:
        logger.debug(f"[{inspect.stack()[0][3]}] Creating OpenAI LLM instance")
        params: dict = {
            "base_url": api_url,
            "openai_api_key": api_key,
            "model": self.model,
            "model_kwargs": {},  # TODO: add model args
            "organization": None,
            "timeout": None,
            "cache": None,
            "streaming": True,
            "temperature": 0.01,
            "max_tokens": 512,
            "top_p": 0.95,
            "frequency_penalty": 1.03,
            "verbose": False,
        }
        # override temperature if defined in developer config
        if config.dev_config.llm_temperature_override:
            params["temperature"] = config.dev_config.llm_temperature_override

        # TODO: We need to verify if the overridden params are valid for the LLM
        # before updating the default.
        # params.update(self.llm_params)  # override parameters
        llm = ChatOpenAI(**params)
        logger.debug(f"[{inspect.stack()[0][3]}] OpenAI LLM instance {llm}")
        return llm

    def _bam_llm_instance(self, api_url: str, api_key: str) -> LLM:
        """BAM Research Lab."""
        logger.debug(f"[{inspect.stack()[0][3]}] BAM LLM instance")
        # BAM Research lab
        creds = Credentials(
            api_key=api_key,
            api_endpoint=api_url,
        )

        bam_params = {
            "decoding_method": "sample",
            "max_new_tokens": 512,
            "min_new_tokens": 1,
            "random_seed": 42,
            "top_k": 10,
            "top_p": 0.95,
            "repetition_penalty": 1.03,
            "temperature": 0.05,
        }
        bam_params.update(self.llm_params)  # override parameters

        # override temperature if defined in developer config
        if config.dev_config.llm_temperature_override:
            bam_params["temperature"] = config.dev_config.llm_temperature_override

        # remove none BAM params from dictionary
        for k in ["model", "api_key", "api_endpoint"]:
            _ = bam_params.pop(k, None)

        client = Client(credentials=creds)
        params = TextGenerationParameters(**bam_params)

        llm = LangChainInterface(client=client, model_id=self.model, parameters=params)
        logger.debug(f"[{inspect.stack()[0][3]}] BAM LLM instance {llm}")
        return llm

    # # TODO: refactor after OLS-233
    # def _ollama_llm_instance(self) -> LLM:
    #     logger.debug(f"[{inspect.stack()[0][3]}] Creating Ollama LLM instance")
    #     try:
    #         from langchain.llms import Ollama
    #     except Exception as e:
    #         logger.error(
    #             "Missing ollama libraries. ollama provider will be unavailable."
    #         )
    #         raise e
    #     params = {
    #         "base_url": os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434"),
    #         "model": os.environ.get("OLLAMA_MODEL", "Mistral"),
    #         "cache": None,
    #         "temperature": 0.01,
    #         "top_k": 10,
    #         "top_p": 0.95,
    #         "repeat_penalty": 1.03,
    #         "verbose": False,
    #         "callback_manager": CallbackManager([StreamingStdOutCallbackHandler()]),
    #     }
    #     params.update(self.llm_params)  # override parameters
    #     llm = Ollama(**params)
    #     logger.debug(f"[{inspect.stack()[0][3]}] Ollama LLM instance {llm}")
    #     return llm

    # # TODO: update this to use config not direct env vars
    # def _tgi_llm_instance(self) -> LLM:
    #     """Note: TGI does not support specifying the model, it is an instance per model."""
    #     logger.debug(
    #         f"[{inspect.stack()[0][3]}] Creating Hugging Face TGI LLM instance"
    #     )
    #     try:
    #         from langchain.llms import HuggingFaceTextGenInference
    #     except Exception as e:
    #         logger.error(
    #             "Missing HuggingFaceTextGenInference libraries. HuggingFaceTextGenInference "
    #             "provider will be unavailable."
    #         )
    #         raise e
    #     params: dict = {
    #         "inference_server_url": os.environ.get("TGI_API_URL", None),
    #         "model_kwargs": {},  # TODO: add model args
    #         "max_new_tokens": 512,
    #         "cache": None,
    #         "temperature": 0.01,
    #         "top_k": 10,
    #         "top_p": 0.95,
    #         "repetition_penalty": 1.03,
    #         "streaming": True,
    #         "verbose": False,
    #         "callback_manager": CallbackManager([StreamingStdOutCallbackHandler()]),
    #     }
    #     params.update(self.llm_params)  # override parameters
    #     llm = HuggingFaceTextGenInference(**params)
    #     logger.debug(
    #         f"[{inspect.stack()[0][3]}] Hugging Face TGI LLM instance {llm}"
    #     )
    #     return llm

    def _watson_llm_instance(self, api_url: str, api_key: str, project_id: str) -> LLM:
        logger.debug(f"[{inspect.stack()[0][3]}] Watson LLM instance")
        # WatsonX uses different keys
        creds = {
            # example from https://heidloff.net/article/watsonx-langchain/
            "url": api_url,
            "apikey": api_key,
        }
        # WatsonX uses different mechanism for defining parameters
        params = {
            GenParams.DECODING_METHOD: self.llm_params.get("decoding_method", "sample"),
            GenParams.MIN_NEW_TOKENS: self.llm_params.get("min_new_tokens", 1),
            GenParams.MAX_NEW_TOKENS: self.llm_params.get("max_new_tokens", 512),
            GenParams.RANDOM_SEED: self.llm_params.get("random_seed", 42),
            GenParams.TEMPERATURE: self.llm_params.get("temperature", 0.05),
            GenParams.TOP_K: self.llm_params.get("top_k", 10),
            GenParams.TOP_P: self.llm_params.get("top_p", 0.95),
            # https://www.ibm.com/docs/en/watsonx-as-a-service?topic=models-parameters
            GenParams.REPETITION_PENALTY: self.llm_params.get(
                "repeatition_penallty", 1.03
            ),
        }

        # override temperature if defined in developer config
        if config.dev_config.llm_temperature_override:
            params[GenParams.TEMPERATURE] = config.dev_config.llm_temperature_override

        # WatsonX uses different parameter names
        llm_model = Model(
            model_id=self.model,
            credentials=creds,
            params=params,
            project_id=project_id,
        )
        llm = WatsonxLLM(model=llm_model)
        logger.debug(f"[{inspect.stack()[0][3]}] Watson LLM instance {llm}")
        return llm

    def status(self):
        """Provide LLM schema as a string containing formatted and indented JSON."""
        import json

        return json.dumps(self.llm.schema_json, indent=4)
