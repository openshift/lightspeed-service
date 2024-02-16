"""Unit tests for LLMLoader class."""

import pytest

from ols import constants
from ols.app.models.config import Config, LLMProviders
from ols.src.llms.llm_loader import (
    LLMConfigurationError,
    LLMLoader,
    ModelConfigInvalidError,
    ModelConfigMissingError,
    UnknownProviderError,
    UnsupportedProviderError,
)
from ols.utils import config


def test_errors_relationship():
    """Test the relationship between LLMConfigurationError and its subclasses."""
    assert issubclass(UnknownProviderError, LLMConfigurationError)
    assert issubclass(UnsupportedProviderError, LLMConfigurationError)
    assert issubclass(ModelConfigMissingError, LLMConfigurationError)
    assert issubclass(ModelConfigInvalidError, LLMConfigurationError)


def test_constructor_unknown_provider():
    """Test raises when provider is unknown."""
    providers = LLMProviders()  # no providers
    config.init_empty_config()
    config.config.llm_providers = providers

    msg = "Provider 'unknown-provider' is not a valid provider"
    with pytest.raises(UnknownProviderError, match=msg):
        LLMLoader(provider="unknown-provider", model="some model")


def test_constructor_unsupported_provider():
    """Test how a configured but an unsupported provider (not openai, bam, etc) is checked for."""
    test_provider = "test-provider"
    test_provider_type = "unsupported-provider-type"
    config.config = Config(
        {
            "llm_providers": [
                {
                    "name": test_provider,
                    "type": "bam",
                    "models": [
                        {
                            "name": constants.GRANITE_13B_CHAT_V1,
                        }
                    ],
                }
            ]
        }
    )
    config.config.llm_providers.providers[test_provider].type = test_provider_type
    config.llm_config = config.config.llm_providers

    with pytest.raises(UnsupportedProviderError):
        LLMLoader(provider=test_provider, model=constants.GRANITE_13B_CHAT_V1)


def test_constructor_unknown_model():
    """Test raise when model configuration is missing."""
    providers = LLMProviders([{"name": "bam", "models": [{"name": "model"}]}])
    config.init_empty_config()
    config.config.llm_providers = providers

    message = "Model 'bla' is not a valid model for provider"
    with pytest.raises(ModelConfigMissingError, match=message):
        LLMLoader(provider="bam", model="bla")


# TODO: As the LLMLoader is a subject of changes via OLS-233. Tests will
# be delivered with the task.
# Once changes are done, refactor also test bellow.
#
# providers = [
#     constants.PROVIDER_OPENAI,
#     constants.PROVIDER_BAM,
# ]
# @pytest.mark.parametrize("provider", providers)
# def test_constructor_no_keys(provider):
#     """Test raise when keys are missing."""
#     test_model = constants.TEI_EMBEDDING_MODEL
#     config.config = Config(
#         {"llm_providers": [{"name": provider, "models": [{"name": test_model}]}]}
#     )
#     config.llm_config = config.config.llm_providers

#     with pytest.raises(ValueError, match="api_key"):
#         LLMLoader(provider=provider, model=test_model)
