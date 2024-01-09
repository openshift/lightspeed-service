"""Unit tests for InMemoryCache class."""

import pytest

from src.cache.in_memory_cache import InMemoryCache


@pytest.fixture
def cache():
    """Fixture with constucted and initialized in memory cache object."""
    cache_size = 10
    c = InMemoryCache(cache_size)
    c.initialize_cache(cache_size)
    return c


def test_insert_or_append(cache):
    """Test the behavior of insert_or_append method."""
    cache.insert_or_append("key1", "value1")
    assert cache.get("key1") == "value1"


def test_insert_or_append_existing_key(cache):
    """Test the behavior of insert_or_append method for existing item."""
    cache.insert_or_append("key1", "value1")
    cache.insert_or_append("key1", "value2")
    assert cache.get("key1") == "value1\nvalue2"


def test_insert_or_append_overflow(cache):
    """Test if items in cache with defined capacity is handled correctly."""
    capacity = 5
    cache.capacity = capacity
    for i in range(capacity + 1):
        key = f"key{i}"
        value = f"value{i}"
        cache.insert_or_append(key, value)

    # Ensure the oldest entry is evicted
    assert cache.get("key0") is None
    # Ensure the newest entry is still present
    assert cache.get(f"key{capacity}") == f"value{capacity}"


def test_get_nonexistent_key(cache):
    """Test how non-existent items are handled by the cache."""
    assert cache.get("nonexistent_key") is None


def test_singleton_pattern():
    """Test if in memory cache exists as one instance in memory."""
    cache1 = InMemoryCache(10)
    cache2 = InMemoryCache(10)
    assert cache1 is cache2
