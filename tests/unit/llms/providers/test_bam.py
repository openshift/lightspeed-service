"""Unit tests for OpenAI provider."""

from genai.extensions.langchain import LangChainInterface

from ols.app.models.config import LLMProviderConfig
from ols.src.llms.providers.bam import BAM
from ols.utils import config


def test_basic_interface():
    """Test basic interface."""
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params
    provider_cfg = LLMProviderConfig(
        **{
            "name": "some_provider",
            "type": "bam",
            "url": "http://test_url.com",
            "credentials_path": "tests/config/secret.txt",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test_url.com",
                    "credentials_path": "tests/config/secret.txt",
                }
            ],
        }
    )

    bam = BAM(model="uber-model", params={}, provider_config=provider_cfg)
    llm = bam.load()
    assert isinstance(llm, LangChainInterface)
    assert bam.default_params
