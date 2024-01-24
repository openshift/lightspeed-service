"""Unit tests for CacheFactory class."""

from unittest.mock import patch

import pytest

from ols import constants
from ols.app.models.config import ConversationCacheConfig
from ols.src.cache.cache_factory import CacheFactory, InMemoryCache, RedisCache
from tests.mock_classes.redis import MockRedis


@pytest.fixture(scope="module")
def in_memory_cache_config():
    """Fixture containing initialized instance of ConversationCacheConfig."""
    return ConversationCacheConfig(
        {
            "type": constants.IN_MEMORY_CACHE,
            constants.IN_MEMORY_CACHE: {"max_entries": 10},
        }
    )


@pytest.fixture(scope="module")
def redis_cache_config():
    """Fixture containing initialized instance of ConversationCacheConfig."""
    return ConversationCacheConfig(
        {
            "type": constants.REDIS_CACHE,
            constants.REDIS_CACHE: {"host": "localhost", "port": 6379},
        }
    )


@pytest.fixture(scope="module")
def invalid_cache_type_config():
    """Fixture containing instance of ConversationCacheConfig with improper settings."""
    c = ConversationCacheConfig()
    c.type = "foo bar baz"
    return c


def test_conversation_cache_in_memory(in_memory_cache_config):
    """Check if InMemoryCache is returned by factory with proper env.var setup."""
    cache = CacheFactory.conversation_cache(in_memory_cache_config)
    assert cache is not None
    # check if the object has the right type
    assert isinstance(cache, InMemoryCache)


@patch("redis.StrictRedis", new=MockRedis)
def test_conversation_cache_in_redis(redis_cache_config):
    """Check if RedisCache is returned by factory with proper env.var setup."""
    cache = CacheFactory.conversation_cache(redis_cache_config)
    assert cache is not None
    # check if the object has the right type
    assert isinstance(cache, RedisCache), type(cache)


def test_conversation_cache_wrong_cache(invalid_cache_type_config):
    """Check if wrong cache env.variable is detected properly."""
    with pytest.raises(ValueError, match="Invalid cache type"):
        CacheFactory.conversation_cache(invalid_cache_type_config)
