"""Unit tests for LLMLoader class."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from ols import constants
from ols.app.models.config import ProviderConfig
from ols.src.llms.llm_loader import (
    LLMConfigurationError,
    LLMLoader,
    MissingModelError,
    MissingProviderError,
    ModelConfigInvalidError,
    ModelConfigMissingError,
    UnsupportedProviderError,
)
from ols.utils import config


def setup():
    """Provide environment for tests."""
    # this function is called automatically by Pytest tool
    # before unit tests are started
    config.init_empty_config()

    # the following modules should not be loaded during unit testing
    # (these are not available on CI anyway)
    mock_modules = [
        "genai",
        "genai.extensions.langchain",
        "genai.text.generation",
        "langchain.llms",
        "ibm_watson_machine_learning.foundation_models",
        "ibm_watson_machine_learning.foundation_models.extensions.langchain",
        "ibm_watson_machine_learning.metanames",
    ]
    # make Python think that the modules are loaded already
    for module in mock_modules:
        sys.modules[module] = MagicMock()


def test_errors_relationship():
    """Test the relationship between LLMConfigurationError and its subclasses."""
    assert issubclass(MissingProviderError, LLMConfigurationError)
    assert issubclass(MissingModelError, LLMConfigurationError)
    assert issubclass(UnsupportedProviderError, LLMConfigurationError)
    assert issubclass(ModelConfigMissingError, LLMConfigurationError)
    assert issubclass(ModelConfigInvalidError, LLMConfigurationError)


def test_constructor_no_provider():
    """Test that constructor checks for provider."""
    with pytest.raises(MissingProviderError, match="Missing provider"):
        LLMLoader(provider=None)


def test_constructor_no_model():
    """Test that constructor checks for model."""
    with pytest.raises(MissingModelError, match="Missing model"):
        LLMLoader(provider=constants.PROVIDER_BAM, model=None)


def test_constructor_wrong_provider():
    """Test how wrong provider is checked."""
    with pytest.raises(UnsupportedProviderError):
        LLMLoader(provider="invalid-provider", model=constants.GRANITE_13B_CHAT_V1)


def test_constructor_unsupported_provider():
    """Test how wrong provider is checked."""
    provider_config = ProviderConfig()
    provider_config.models = {constants.GRANITE_13B_CHAT_V1: "mock-model"}
    config.llm_config.providers = {"unsupported-provider": provider_config}
    with pytest.raises(UnsupportedProviderError):
        LLMLoader(provider="unsupported-provider", model=constants.GRANITE_13B_CHAT_V1)


def test_constructor_when_missing_model():
    """Test raise when no model is provided."""
    test_provider = "test-provider"
    test_model = "test-model"
    provider_config = ProviderConfig()
    provider_config.models = {}  # no models configured
    config.llm_config.providers = {test_provider: provider_config}
    message = f"No configuration provided for model {test_model} under LLM provider {test_provider}"
    # ModelConfigMissingException instead of KeyError is raised
    with pytest.raises(ModelConfigMissingError, match=message):
        LLMLoader(provider=test_provider, model=test_model)


def test_constructor_when_missing_model_config():
    """Test raise when model configuration is missing."""
    test_provider = "test-provider"
    test_model = "test-model"
    provider_config = ProviderConfig()
    provider_config.models = {
        test_model: None
    }  # model configured but no config provided
    config.llm_config.providers = {test_provider: provider_config}
    message = f"No configuration provided for model {test_model} under LLM provider {test_provider}"
    with pytest.raises(ModelConfigMissingError, match=message):
        LLMLoader(provider=test_provider, model=test_model)


# all LLM providers that can be initialized
llm_providers = [
    constants.PROVIDER_OPENAI,
    # constants.PROVIDER_OLLAMA,
    constants.PROVIDER_WATSONX,
    # constants.PROVIDER_TGI,
    constants.PROVIDER_BAM,
]


@pytest.mark.parametrize("provider", llm_providers)
def test_constructor_unsatisfied_requirements(provider):
    """Test how unsatisfied requirements are handled by LLM loader."""
    provider_config = ProviderConfig()
    provider_config.models = {constants.GRANITE_13B_CHAT_V1: "mock model"}
    config.llm_config.providers = {provider: provider_config}

    def mock_import(module, *args, **kwargs):
        """Mock the import and from x import statements."""

    # check what happens if LLM libraries can not be loaded
    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="cannot import name"):
            LLMLoader(provider=provider, model=constants.GRANITE_13B_CHAT_V1)


def test_constructor_openai_llm_instance_no_api_key():
    """Test the construction fo LLM instance for OpenAI when API key is not provided."""
    # no API key is provided so validation should fail
    config.init_config("tests/config/without_openai_api_key.yaml")
    with pytest.raises(Exception, match="Did not find openai_api_key"):
        LLMLoader(provider=constants.PROVIDER_OPENAI, model=constants.GPT35_TURBO)


def test_constructor_openai_llm_instance_provided_api_key():
    """Test the construction fo LLM instance for OpenAI when API key is provided."""
    config.init_config("tests/config/with_openai_api_key.yaml")
    # API key is provided so validation must not fail
    LLMLoader(provider=constants.PROVIDER_OPENAI, model=constants.GPT35_TURBO)
