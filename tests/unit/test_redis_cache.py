"""Unit tests for RedisCache class."""

from unittest.mock import patch

import pytest

from ols.app.utils import Utils
from ols.src.cache.redis_cache import RedisCache
from tests.mock_classes.redis import MockRedis

conversation_id = Utils.get_suid()


@pytest.fixture
def cache():
    """Fixture with constucted and initialized Redis cache object."""
    # we don't want to connect to real Redis from unit tests
    # with patch("ols.src.cache.redis_cache.RedisCache.initialize_redis"):
    with patch("redis.StrictRedis", new=MockRedis):
        return RedisCache()


def test_insert_or_append(cache):
    """Test the behavior of insert_or_append method."""
    assert cache.get("user1", conversation_id) is None
    cache.insert_or_append("user1", conversation_id, "value1")
    assert cache.get("user1", conversation_id) == "value1"


def test_insert_or_append_existing_key(cache):
    """Test the behavior of insert_or_append method for existing item."""
    # conversation IDs are separated by users
    assert cache.get("user2", conversation_id) is None

    cache.insert_or_append("user2", conversation_id, "value1")
    cache.insert_or_append("user2", conversation_id, "value2")
    assert cache.get("user2", conversation_id) == "value1\nvalue2"


def test_get_nonexistent_key(cache):
    """Test how non-existent items are handled by the cache."""
    assert cache.get("nonexistent_key", conversation_id) is None


def test_get_improper_user_id(cache):
    """Test how improper user ID is handled."""
    with pytest.raises(ValueError):
        assert cache.get("foo/bar", conversation_id) is None


def test_get_improper_conversation_id(cache):
    """Test how improper conversation ID is handled."""
    with pytest.raises(ValueError):
        assert cache.get("user1", "this-is-not-valid-uuid") is None


def test_singleton_pattern():
    """Test if in memory cache exists as one instance in memory."""
    cache1 = RedisCache()
    cache2 = RedisCache()
    assert cache1 is cache2
