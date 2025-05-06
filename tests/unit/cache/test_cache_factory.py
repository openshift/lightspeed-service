"""Unit tests for CacheFactory class."""

from unittest.mock import patch

import pytest

from ols import constants
from ols.app.models.config import ConversationCacheConfig
from ols.src.cache.cache_factory import (
    CacheFactory,
    InMemoryCache,
    PostgresCache,
)


@pytest.fixture(scope="module")
def in_memory_cache_config():
    """Fixture containing initialized instance of ConversationCacheConfig."""
    return ConversationCacheConfig(
        {
            "type": constants.CACHE_TYPE_MEMORY,
            constants.CACHE_TYPE_MEMORY: {"max_entries": 10},
        }
    )


@pytest.fixture(scope="module")
def postgres_cache_config():
    """Fixture containing initialized instance of ConversationCacheConfig."""
    return ConversationCacheConfig(
        {
            "type": constants.CACHE_TYPE_POSTGRES,
            constants.CACHE_TYPE_POSTGRES: {"host": "localhost", "port": 5432},
        }
    )


@pytest.fixture(scope="module")
def invalid_cache_type_config():
    """Fixture containing instance of ConversationCacheConfig with improper settings."""
    c = ConversationCacheConfig()
    c.type = "foo bar baz"
    return c


def test_conversation_cache_in_memory(in_memory_cache_config):
    """Check if InMemoryCache is returned by factory with proper configuration."""
    cache = CacheFactory.conversation_cache(in_memory_cache_config)
    assert cache is not None
    # check if the object has the right type
    assert isinstance(cache, InMemoryCache)


def test_conversation_cache_in_postgres(postgres_cache_config):
    """Check if PostgresCache is returned by factory with proper configuration."""
    # do not use real PostgreSQL instance
    with patch("psycopg2.connect"):
        cache = CacheFactory.conversation_cache(postgres_cache_config)

    assert cache is not None
    # check if the object has the right type
    assert isinstance(cache, PostgresCache), type(cache)


def test_conversation_cache_wrong_cache(invalid_cache_type_config):
    """Check if wrong cache configuration is detected properly."""
    with pytest.raises(ValueError, match="Invalid cache type"):
        CacheFactory.conversation_cache(invalid_cache_type_config)
