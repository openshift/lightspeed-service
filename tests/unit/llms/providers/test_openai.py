"""Unit tests for OpenAI provider."""

from langchain_openai.chat_models.base import ChatOpenAI

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.openai import OpenAI
from ols.utils import config


def test_basic_interface():
    """Test basic interface."""
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params
    provider_cfg = ProviderConfig(
        {
            "name": "some_provider",
            "type": "openai",
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

    openai = OpenAI(model="uber-model", params={}, provider_config=provider_cfg)
    llm = openai.load()
    assert isinstance(llm, ChatOpenAI)
    assert openai.default_params
