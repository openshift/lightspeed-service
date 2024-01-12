"""Unit tests for RedisCache class."""

from unittest.mock import patch

import pytest

from ols.src.cache.redis_cache import RedisCache
from tests.mock_classes.redis import MockRedis


@pytest.fixture
def cache():
    """Fixture with constucted and initialized Redis cache object."""
    # we don't want to connect to real Redis from unit tests
    # with patch("ols.src.cache.redis_cache.RedisCache.initialize_redis"):
    with patch("redis.StrictRedis", new=MockRedis):
        return RedisCache()


def test_insert_or_append(cache):
    """Test the behavior of insert_or_append method."""
    assert cache.get("key1") is None
    cache.insert_or_append("key1", "value1")
    assert cache.get("key1") == "value1"


def test_insert_or_append_existing_key(cache):
    """Test the behavior of insert_or_append method for existing item."""
    assert cache.get("key2") is None

    cache.insert_or_append("key2", "value1")
    cache.insert_or_append("key2", "value2")
    assert cache.get("key2") == "value1\nvalue2"


def test_get_nonexistent_key(cache):
    """Test how non-existent items are handled by the cache."""
    assert cache.get("nonexistent_key") is None


def test_singleton_pattern():
    """Test if in memory cache exists as one instance in memory."""
    cache1 = RedisCache()
    cache2 = RedisCache()
    assert cache1 is cache2
