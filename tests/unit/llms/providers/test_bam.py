"""Unit tests for OpenAI provider."""

from genai.extensions.langchain import LangChainInterface

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.bam import BAM
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

    openai = BAM(model="uber-model", params={}, provider_config=provider_cfg)
    llm = openai.load()
    assert isinstance(llm, LangChainInterface)
    assert openai.default_params
