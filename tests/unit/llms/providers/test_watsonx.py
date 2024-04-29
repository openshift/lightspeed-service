"""Unit tests for Watsonx provider."""

from unittest.mock import patch

import pytest

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.watsonx import WatsonX
from ols.utils import config
from tests.mock_classes.mock_watsonxllm import WatsonxLLM


@pytest.fixture
def provider_config():
    """Fixture with provider configuration for Watsonx."""
    return ProviderConfig(
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


@patch("ols.src.llms.providers.watsonx.WatsonxLLM", new=WatsonxLLM())
def test_basic_interface(provider_config):
    """Test basic interface."""
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params

    watsonx = WatsonX(model="uber-model", params={}, provider_config=provider_config)
    llm = watsonx.load()
    assert isinstance(llm, WatsonxLLM)
    assert watsonx.default_params
