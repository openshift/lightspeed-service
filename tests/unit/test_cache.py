import pytest

from lightspeed_service.cache.in_memory_cache import InMemoryCache


@pytest.fixture
def cache():
    c = InMemoryCache()
    c.initialize_cache()
    return c


def test_insert_or_append(cache):
    cache.insert_or_append("key1", "value1")
    assert cache.get("key1") == "value1"


def test_insert_or_append_existing_key(cache):
    cache.insert_or_append("key1", "value1")
    cache.insert_or_append("key1", "value2")
    assert cache.get("key1") == "value1\nvalue2"


def test_insert_or_append_overflow(cache):
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
    assert cache.get("nonexistent_key") is None


def test_singleton_pattern():
    cache1 = InMemoryCache()
    cache2 = InMemoryCache()
    assert cache1 is cache2
