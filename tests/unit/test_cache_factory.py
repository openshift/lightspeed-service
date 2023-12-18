import os
import pytest

import src.constants as constants
from src.cache.cache_factory import CacheFactory
from src.cache.cache_factory import InMemoryCache


@pytest.fixture(scope="module")
def in_memory_cache_env_var():
    os.environ["OLS_CONVERSATION_CACHE"] = constants.IN_MEMORY_CACHE


@pytest.fixture(scope="module")
def wrong_cache_env_var():
    os.environ["OLS_CONVERSATION_CACHE"] = "foo bar baz"


def test_conversation_cache(in_memory_cache_env_var):
    """Check if InMemoryCache is returned by factory with proper env.var setup."""
    cache = CacheFactory.conversation_cache()
    assert cache is not None
    # check if the object has the right type
    assert isinstance(cache, InMemoryCache)


def test_conversation_cache_wrong_cache(wrong_cache_env_var):
    """Check if wrong cache env.variable is detected properly."""
    with pytest.raises(ValueError):
        CacheFactory.conversation_cache()
