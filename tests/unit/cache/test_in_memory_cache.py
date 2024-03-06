"""Unit tests for InMemoryCache class."""

import pytest

from ols import constants
from ols.app.models.config import MemoryConfig
from ols.src.cache.in_memory_cache import InMemoryCache
from ols.utils import suid

conversation_id = suid.get_suid()


@pytest.fixture
def cache():
    """Fixture with constucted and initialized in memory cache object."""
    mc = MemoryConfig({"max_entries": "10"})
    c = InMemoryCache(mc)
    c.initialize_cache(mc)
    return c


def test_insert_or_append(cache):
    """Test the behavior of insert_or_append method."""
    cache.insert_or_append(constants.DEFAULT_USER_UID, conversation_id, "value1")
    assert cache.get(constants.DEFAULT_USER_UID, conversation_id) == "value1"


def test_insert_or_append_existing_key(cache):
    """Test the behavior of insert_or_append method for existing item."""
    cache.insert_or_append(constants.DEFAULT_USER_UID, conversation_id, "value1")
    cache.insert_or_append(constants.DEFAULT_USER_UID, conversation_id, "value2")
    assert cache.get(constants.DEFAULT_USER_UID, conversation_id) == "value1\nvalue2"


def test_insert_or_append_overflow(cache):
    """Test if items in cache with defined capacity is handled correctly."""
    # remove last hex digit from user UUID
    user_name_prefix = constants.DEFAULT_USER_UID[:-1]

    capacity = 5
    cache.capacity = capacity
    for i in range(capacity + 1):
        user = f"{user_name_prefix}{i}"
        value = f"value{i}"
        cache.insert_or_append(user, conversation_id, value)

    # Ensure the oldest entry is evicted
    assert cache.get(f"{user_name_prefix}0", conversation_id) is None
    # Ensure the newest entry is still present
    assert (
        cache.get(f"{user_name_prefix}{capacity}", conversation_id)
        == f"value{capacity}"
    )


def test_get_nonexistent_user(cache):
    """Test how non-existent items are handled by the cache."""
    # this UUID is different from DEFAULT_USER_UID
    assert cache.get("ffffffff-ffff-ffff-ffff-ffffffffffff", conversation_id) is None


improper_user_uuids = [
    "",
    " ",
    "\t",
    ":",
    "foo:bar",
    "ffffffff-ffff-ffff-ffff-fffffffffff",  # UUID-like string with missing chararacter
    "ffffffff-ffff-ffff-ffff-fffffffffffZ",  # UUID-like string, but with wrong character
    "ffffffff:ffff:ffff:ffff:ffffffffffff",
]


@pytest.mark.parametrize("uuid", improper_user_uuids)
def test_get_improper_user_id(cache, uuid):
    """Test how improper user ID is handled."""
    with pytest.raises(ValueError, match=f"Invalid user ID {uuid}"):
        cache.get(uuid, conversation_id)


def test_get_improper_conversation_id(cache):
    """Test how improper conversation ID is handled."""
    with pytest.raises(ValueError, match="Invalid conversation ID"):
        cache.get(constants.DEFAULT_USER_UID, "this-is-not-valid-uuid")


def test_singleton_pattern():
    """Test if in memory cache exists as one instance in memory."""
    mc = MemoryConfig({"max_entries": "10"})
    cache1 = InMemoryCache(mc)
    cache2 = InMemoryCache(mc)
    assert cache1 is cache2
