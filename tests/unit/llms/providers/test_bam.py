"""Unit tests for BAM provider."""

import pytest
from genai.extensions.langchain import LangChainInterface

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.bam import BAM
from ols.utils import config


@pytest.fixture
def provider_config():
    """Fixture with provider configuration for BAM."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "bam",
            "url": "test_url",
            "credentials_path": "tests/config/secret.txt",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "test_model_url",
                    "credentials_path": "tests/config/secret.txt",
                }
            ],
        }
    )


def test_basic_interface(provider_config):
    """Test basic interface."""
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params

    bam = BAM(model="uber-model", params={}, provider_config=provider_config)
    llm = bam.load()
    assert isinstance(llm, LangChainInterface)
    assert bam.default_params
