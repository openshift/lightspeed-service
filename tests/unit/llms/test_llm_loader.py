"""Unit tests for LLMLoader class."""

import sys
from unittest.mock import MagicMock

import pytest

from ols import constants
from ols.app.models.config import Config
from ols.src.llms.llm_loader import (
    LLMConfigurationError,
    LLMLoader,
    MissingModelError,
    MissingProviderError,
    ModelConfigInvalidError,
    ModelConfigMissingError,
    UnknownProviderError,
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
    assert issubclass(UnknownProviderError, LLMConfigurationError)
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


def test_constructor_unknown_provider():
    """Test how a provider missing from configuration is checked for."""
    with pytest.raises(UnknownProviderError):
        LLMLoader(provider="unknown-provider", model=constants.GRANITE_13B_CHAT_V1)


def test_constructor_unsupported_provider():
    """Test how a configured but an unsupported provider (not openai, bam, etc) is checked for."""
    test_provider = "unsupported-provider"
    config.config = Config(
        {
            "llm_providers": [
                {
                    "name": test_provider,
                    "models": [
                        {
                            "name": constants.GRANITE_13B_CHAT_V1,
                        }
                    ],
                }
            ]
        }
    )
    config.llm_config = config.config.llm_providers

    with pytest.raises(UnsupportedProviderError):
        LLMLoader(provider=test_provider, model=constants.GRANITE_13B_CHAT_V1)


def test_constructor_when_missing_model_config():
    """Test raise when model configuration is missing."""
    test_provider = "test-provider"
    test_model = "test-model"
    config.config = Config(
        {"llm_providers": [{"name": test_provider, "models": [{"name": "foobar"}]}]}
    )
    config.llm_config = config.config.llm_providers

    message = f"No configuration provided for model {test_model} under LLM provider {test_provider}"
    with pytest.raises(ModelConfigMissingError, match=message):
        LLMLoader(provider=test_provider, model=test_model)
