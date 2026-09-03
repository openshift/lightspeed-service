"""Watsonx provider implementation."""

import logging
from typing import Any, Optional
from urllib.parse import urlparse

from ibm_watsonx_ai.metanames import (
    GenTextParamsMetaNames as GenParams,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ibm import ChatWatsonx

from ols import constants
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)

IBM_CLOUD_WATSONX_HOST_SUFFIX = "ml.cloud.ibm.com"


def is_ibm_cloud_watsonx_url(url: str) -> bool:
    """Return True when the URL is IBM Cloud watsonx SaaS (apitoken only)."""
    if not url:
        return True
    host = (urlparse(str(url)).hostname or "").lower()
    return host == IBM_CLOUD_WATSONX_HOST_SUFFIX or host.endswith(
        "." + IBM_CLOUD_WATSONX_HOST_SUFFIX
    )


@register_llm_provider_as(constants.PROVIDER_WATSONX)
class Watsonx(LLMProvider):
    """Watsonx provider."""

    url: str = "https://us-south.ml.cloud.ibm.com"
    credentials: Optional[str]
    project_id: Optional[str]

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return structure with default LLM params."""
        # https://www.ibm.com/docs/en/watsonx-as-a-service?topic=models-parameters
        return {
            GenParams.DECODING_METHOD: "sample",
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 512,
            GenParams.RANDOM_SEED: 42,
            GenParams.TEMPERATURE: 0.05,
            GenParams.TOP_K: 50,
            GenParams.TOP_P: 0.85,
            GenParams.REPETITION_PENALTY: 1.05,
        }

    def load(self) -> BaseChatModel:
        """Load LLM."""
        self.url = str(self.provider_config.url or self.url)
        self.credentials = self.provider_config.credentials
        self.project_id = self.provider_config.project_id

        # provider-specific configuration has precendence over regular configuration
        if self.provider_config.watsonx_config is not None:
            watsonx_config = self.provider_config.watsonx_config
            self.url = str(watsonx_config.url)
            self.project_id = watsonx_config.project_id
            if watsonx_config.api_key is not None:
                self.credentials = watsonx_config.api_key

        if self.credentials is None:
            raise ValueError("Credentials must be specified")

        if self.project_id is None:
            raise ValueError("Project ID must be specified")

        constructor_kwargs: dict[str, Any] = {
            "model_id": self.model,
            "url": self.url,
            "apikey": self.credentials,
            "project_id": self.project_id,
            "params": self.params,
        }

        if not is_ibm_cloud_watsonx_url(self.url):
            username = self.provider_config.watsonx_username
            version = self.provider_config.watsonx_version
            missing = [
                name
                for name, value in (("username", username), ("version", version))
                if not value
            ]
            if missing:
                needed = " and ".join(missing)
                raise ValueError(
                    "On-prem Cloud Pak for Data watsonx needs "
                    f"{needed} in the credentials secret (keys next to apitoken). "
                    "IBM Cloud watsonx still only needs apitoken."
                )
            constructor_kwargs["username"] = username
            constructor_kwargs["version"] = version
            constructor_kwargs["instance_id"] = (
                self.provider_config.watsonx_instance_id
                or constants.WATSONX_DEFAULT_CPD_INSTANCE_ID
            )

        logger.info(
            "Loading WatsonX LLM: model=%s, url=%s, project_id=%s",
            self.model,
            self.url,
            self.project_id,
        )
        return ChatWatsonx(**constructor_kwargs)
