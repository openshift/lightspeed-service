"""Unit tests for LLMLoader class."""

import pytest

from ols.src import constants
from ols.src.llms.llm_loader import LLMLoader, UnsupportedProvider


def test_constructor_no_provider():
    """Test that constructor checks for provider."""
    # we use bare Exception in the code, so need to check
    # message, at least
    with pytest.raises(Exception, match="ERROR: Missing provider"):
        LLMLoader(provider=None)


def test_constructor_wrong_provider():
    """Test how wrong provider is checked."""
    # currently, it just logs error and that's it
    with pytest.raises(UnsupportedProvider):
        LLMLoader(provider="invalid-provider", model=constants.GRANITE_13B_CHAT_V1)
