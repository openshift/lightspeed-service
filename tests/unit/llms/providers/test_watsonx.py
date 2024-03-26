"""Unit tests for OpenAI provider."""

from unittest.mock import MagicMock, patch

from ibm_watson_machine_learning.foundation_models.extensions.langchain import (
    WatsonxLLM,
)

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.watsonx import WatsonX
from ols.utils import config


@patch("ols.src.llms.providers.watsonx.Model", new=MagicMock())
def test_basic_interface():
    """Test basic interface."""
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params
    provider_cfg = ProviderConfig(
        {
            "name": "some_provider",
            "type": "watsonx",
            "url": "https://us-south.ml.cloud.ibm.com",
            "credentials_path": "tests/config/secret.txt",
            "project_id": "01234567-89ab-cdef-0123-456789abcdef",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "test_model_url",
                    "credentials_path": "tests/config/secret.txt",
                }
            ],
        }
    )

    openai = WatsonX(model="uber-model", params={}, provider_config=provider_cfg)
    llm = openai.load()
    assert isinstance(llm, WatsonxLLM)
    assert openai.default_params
