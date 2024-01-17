"""LLM backend libraries loader."""

import inspect
import os
import warnings
from typing import Optional

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from ols import constants
from ols.utils import config
from ols.utils.logger import Logger

# workaround to disable UserWarning
warnings.simplefilter("ignore", UserWarning)


class MissingProvider(Exception):
    """Provider is not specified."""


class MissingModel(Exception):
    """Model is not specified."""


class UnsupportedProvider(Exception):
    """Provider is not supported or is unknown."""


class ModelConfigMissingException(Exception):
    """No configuration exists for the requested model name."""


class ModelConfigInvalidException(Exception):
    """Model configuration is not valid."""


class LLMLoader:
    """Note: This class loads the LLM backend libraries if the specific LLM is loaded.

    Known caveats: Currently supports a single instance/model per backend.

    llm_backends: a string with a supported llm backend name ('openai','ollama','tgi','watson','bam').
    params      : (optional) array of parameters to override and pass to the llm backend

    # using the class and overriding specific parameters
    llm_backend = 'ollama'
    params = {'temperature': 0.02, 'top_p': 0.95}

    llm_config = LLMLoader(llm_backend=llm_backend, params=params)
    llm_chain = LLMChain(llm=llm_config.llm, prompt=prompt)

    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        url: Optional[str] = None,
        params: Optional[dict] = None,
        logger=None,
    ) -> None:
        """Initialize loader using provided provider, model, and other parameters."""
        self.logger = logger if logger is not None else Logger("llm_loader").logger
        if provider is None:
            msg = "Missing provider"
            self.logger.error(msg)
            raise MissingProvider(msg)
        self.provider = provider
        self.url = url
        if model is None:
            msg = "Missing model"
            self.logger.error(msg)
            raise MissingModel(msg)
        self.model = model

        # return empty dictionary if not defined
        self.llm_params = params if params else {}
        self.llm = None
        self._set_llm_instance()

    def _set_llm_instance(self):
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] Loading LLM {self.model} from {self.provider}"
        )
        # convert to string to handle None or False definitions
        match str(self.provider).lower():
            case constants.PROVIDER_OPENAI:
                self._openai_llm_instance()
            case constants.PROVIDER_OLLAMA:
                self._ollama_llm_instance()
            case constants.PROVIDER_TGI:
                self._tgi_llm_instance()
            case constants.PROVIDER_WATSONX:
                self._watson_llm_instance()
            case constants.PROVIDER_BAM:
                self._bam_llm_instance()
            case _:
                msg = f"Unsupported LLM provider {self.provider}"
                self.logger.error(msg)
                raise UnsupportedProvider(msg)

    def _openai_llm_instance(self):
        self.logger.debug(f"[{inspect.stack()[0][3]}] Creating OpenAI LLM instance")
        try:
            from langchain.chat_models import ChatOpenAI
        except Exception as e:
            self.logger.error(
                "Missing openai libraries. Openai provider will be unavailable."
            )
            raise e
        provider = config.llm_config.providers[constants.PROVIDER_OPENAI]
        model = provider.models[self.model]
        if model is None:
            msg = f"No configuration provided for model {self.model} under LLM provider {constants.PROVIDER_OPENAI}"
            self.logger.error(msg)
            raise ModelConfigMissingException(msg)
        params = {
            "base_url": provider.url
            if provider.url is not None
            else "https://api.openai.com/v1",
            "api_key": provider.credentials,
            "model": self.model,
            "model_kwargs": {},  # TODO: add model args
            "organization": os.environ.get("OPENAI_ORGANIZATION", None),
            "timeout": os.environ.get("OPENAI_TIMEOUT", None),
            "cache": None,
            "streaming": True,
            "temperature": 0.01,
            "max_tokens": 512,
            "top_p": 0.95,
            "frequency_penalty": 1.03,
            "verbose": False,
        }
        # TODO: We need to verify if the overridden params are valid for the LLM
        # before updating the default.
        # params.update(self.llm_params)  # override parameters
        self.llm = ChatOpenAI(**params)
        self.logger.debug(f"[{inspect.stack()[0][3]}] OpenAI LLM instance {self.llm}")

    def _bam_llm_instance(self):
        """BAM Research Lab."""
        self.logger.debug(f"[{inspect.stack()[0][3]}] BAM LLM instance")
        try:
            # BAM Research lab
            from genai.credentials import Credentials
            from genai.extensions.langchain import LangChainInterface
            from genai.schemas import GenerateParams
        except Exception as e:
            self.logger.error(
                "Missing ibm-generative-ai libraries. ibm-generative-ai provider will be unavailable."
            )
            raise e
        # BAM Research lab
        provider = config.llm_config.providers[constants.PROVIDER_BAM]
        model = provider.models[self.model]
        if model is None:
            msg = f"No configuration provided for model {self.model} under LLM provider {constants.PROVIDER_BAM}"
            self.logger.error(msg)
            raise ModelConfigMissingException(msg)

        creds = Credentials(
            api_key=provider.credentials,
            api_endpoint=provider.url
            if provider.url is not None
            else "https://bam-api.res.ibm.com",
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
        # remove none BAM params from dictionary
        for k in ["model", "api_key", "api_endpoint"]:
            _ = bam_params.pop(k, None)
        params = GenerateParams(**bam_params)

        self.llm = LangChainInterface(
            model=self.model, params=params, credentials=creds
        )
        self.logger.debug(f"[{inspect.stack()[0][3]}] BAM LLM instance {self.llm}")

    # TODO: update this to use config not direct env vars
    def _ollama_llm_instance(self):
        self.logger.debug(f"[{inspect.stack()[0][3]}] Creating Ollama LLM instance")
        try:
            from langchain.llms import Ollama
        except Exception as e:
            self.logger.error(
                "Missing ollama libraries. ollama provider will be unavailable."
            )
            raise e
        params = {
            "base_url": os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434"),
            "model": os.environ.get("OLLAMA_MODEL", "Mistral"),
            "cache": None,
            "temperature": 0.01,
            "top_k": 10,
            "top_p": 0.95,
            "repeat_penalty": 1.03,
            "verbose": False,
            "callback_manager": CallbackManager([StreamingStdOutCallbackHandler()]),
        }
        params.update(self.llm_params)  # override parameters
        self.llm = Ollama(**params)
        self.logger.debug(f"[{inspect.stack()[0][3]}] Ollama LLM instance {self.llm}")

    # TODO: update this to use config not direct env vars
    def _tgi_llm_instance(self):
        """Note: TGI does not support specifying the model, it is an instance per model."""
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] Creating Hugging Face TGI LLM instance"
        )
        try:
            from langchain.llms import HuggingFaceTextGenInference
        except Exception as e:
            self.logger.error(
                "Missing HuggingFaceTextGenInference libraries. HuggingFaceTextGenInference provider will be unavailable."
            )
            raise e
        params = {
            "inference_server_url": os.environ.get("TGI_API_URL", None),
            "model_kwargs": {},  # TODO: add model args
            "max_new_tokens": 512,
            "cache": None,
            "temperature": 0.01,
            "top_k": 10,
            "top_p": 0.95,
            "repetition_penalty": 1.03,
            "streaming": True,
            "verbose": False,
            "callback_manager": CallbackManager([StreamingStdOutCallbackHandler()]),
        }
        params.update(self.llm_params)  # override parameters
        self.llm = HuggingFaceTextGenInference(**params)
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] Hugging Face TGI LLM instance {self.llm}"
        )

    # TODO: update this to use config not direct env vars
    def _watson_llm_instance(self):
        self.logger.debug(f"[{inspect.stack()[0][3]}] Watson LLM instance")
        # WatsonX (requires WansonX libraries)
        try:
            from ibm_watson_machine_learning.foundation_models import Model
            from ibm_watson_machine_learning.foundation_models.extensions.langchain import (
                WatsonxLLM,
            )
            from ibm_watson_machine_learning.metanames import (
                GenTextParamsMetaNames as GenParams,
            )
        except Exception as e:
            self.logger.error(
                "Missing ibm_watson_machine_learning libraries. Skipping loading backend LLM."
            )
            raise e
        # WatsonX uses different keys
        creds = {
            # example from https://heidloff.net/article/watsonx-langchain/
            "url": self.llm_params.get("url")
            if self.llm_params.get("url") is not None
            else os.environ.get("WATSON_API_URL", None),
            "apikey": self.llm_params.get("apikey")
            if self.llm_params.get("apikey") is not None
            else os.environ.get("WATSON_API_KEY", None),
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
        # WatsonX uses different parameter names
        llm_model = Model(
            model_id=self.llm_params.get(
                "model_id", os.environ.get("WATSON_MODEL", None)
            ),
            credentials=creds,
            params=params,
            project_id=self.llm_params.get(
                "project_id", os.environ.get("WATSON_PROJECT_ID", None)
            ),
        )
        self.llm = WatsonxLLM(model=llm_model)
        self.logger.debug(f"[{inspect.stack()[0][3]}] Watson LLM instance {self.llm}")

    def status(self):
        """Provide LLM schema as a string containing formatted and indented JSON."""
        import json

        return json.dumps(self.llm.schema_json, indent=4)
