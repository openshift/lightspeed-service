"""Unit tests for InMemoryCache class."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ols import constants
from ols.app.models.config import InMemoryCacheConfig
from ols.app.models.models import CacheEntry
from ols.src.cache.in_memory_cache import InMemoryCache
from ols.utils import suid

conversation_id = suid.get_suid()
user_provided_user_id = "test-user1"
cache_entry_1 = CacheEntry(
    query=HumanMessage("user message1"), response=AIMessage("ai message1")
)
cache_entry_2 = CacheEntry(
    query=HumanMessage("user message2"), response=AIMessage("ai message2")
)


@pytest.fixture
def cache():
    """Fixture with constucted and initialized in memory cache object."""
    mc = InMemoryCacheConfig({"max_entries": "10"})
    c = InMemoryCache(mc)
    c.initialize_cache(mc)
    return c


def test_insert_or_append(cache):
    """Test the behavior of insert_or_append method."""
    cache.insert_or_append(
        constants.DEFAULT_USER_UID,
        conversation_id,
        cache_entry_1,
    )

    assert cache.get(constants.DEFAULT_USER_UID, conversation_id) == [cache_entry_1]


def test_insert_or_append_skip_user_id_check(cache):
    """Test the behavior of insert_or_append method."""
    skip_user_id_check = True
    cache.insert_or_append(
        user_provided_user_id, conversation_id, cache_entry_1, skip_user_id_check
    )

    assert cache.get(user_provided_user_id, conversation_id, skip_user_id_check) == [
        cache_entry_1
    ]


def test_insert_or_append_existing_key(cache):
    """Test the behavior of insert_or_append method for existing item."""
    cache.insert_or_append(
        constants.DEFAULT_USER_UID,
        conversation_id,
        cache_entry_1,
    )
    cache.insert_or_append(
        constants.DEFAULT_USER_UID,
        conversation_id,
        cache_entry_2,
    )
    expected_cache = [
        cache_entry_1,
        cache_entry_2,
    ]

    assert cache.get(constants.DEFAULT_USER_UID, conversation_id) == expected_cache


def test_insert_or_append_overflow(cache):
    """Test if items in cache with defined capacity is handled correctly."""
    # remove last hex digit from user UUID
    user_name_prefix = constants.DEFAULT_USER_UID[:-1]

    capacity = 5
    cache.capacity = capacity
    for i in range(capacity + 1):
        user = f"{user_name_prefix}{i}"
        value = CacheEntry(query=HumanMessage(f"user query {i}"))
        cache.insert_or_append(
            user,
            conversation_id,
            value,
        )

    # Ensure the oldest entry is evicted
    assert cache.get(f"{user_name_prefix}0", conversation_id) is None
    # Ensure the newest entry is still present
    expected_result = [CacheEntry(query=HumanMessage(f"user query {i}"))]
    assert (
        cache.get(f"{user_name_prefix}{capacity}", conversation_id) == expected_result
    )


def test_insert_or_append_eviction(cache):
    """Test if items in cache eviction with defined capacity is handled correctly."""
    # remove last hex digit from user UUID
    user_name_prefix = constants.DEFAULT_USER_UID[:-1]

    capacity = 5
    cache.capacity = capacity
    cache.insert_or_append(
        f"{user_name_prefix}{0}",
        conversation_id,
        CacheEntry(query=HumanMessage("user query 0 intial entry")),
    )
    for i in range(capacity):
        user = f"{user_name_prefix}{i}"
        value = CacheEntry(query=HumanMessage(f"user query {i}"))
        cache.insert_or_append(
            user,
            conversation_id,
            value,
        )

    # Ensure the oldest entry is evicted
    expected_result = [CacheEntry(query=HumanMessage("user query 0"))]
    assert cache.get(f"{user_name_prefix}0", conversation_id) == expected_result
    # Ensure the newest entry is still present
    expected_result = [CacheEntry(query=HumanMessage(f"user query {i}"))]
    assert (
        cache.get(f"{user_name_prefix}{capacity - 1}", conversation_id)
        == expected_result
    )


def test_get_nonexistent_user(cache):
    """Test how non-existent items are handled by the cache."""
    # this UUID is different from DEFAULT_USER_UID
    assert cache.get("ffffffff-ffff-ffff-ffff-ffffffffffff", conversation_id) is None


def test_delete_existing_conversation(cache):
    """Test deleting an existing conversation."""
    cache.insert_or_append(constants.DEFAULT_USER_UID, conversation_id, cache_entry_1)

    result = cache.delete(constants.DEFAULT_USER_UID, conversation_id)

    assert result is True
    assert cache.get(constants.DEFAULT_USER_UID, conversation_id) is None


def test_delete_nonexistent_conversation(cache):
    """Test deleting a conversation that doesn't exist."""
    result = cache.delete(constants.DEFAULT_USER_UID, conversation_id)
    assert result is False


def test_delete_improper_conversation_id(cache):
    """Test delete with invalid conversation ID."""
    with pytest.raises(ValueError, match="Invalid conversation ID"):
        cache.delete(constants.DEFAULT_USER_UID, "invalid-id")


def test_delete_skip_user_id_check(cache):
    """Test deleting an existing conversation."""
    skip_user_id_check = True
    cache.insert_or_append(
        user_provided_user_id, conversation_id, cache_entry_1, skip_user_id_check
    )

    result = cache.delete(user_provided_user_id, conversation_id, skip_user_id_check)

    assert result is True
    assert cache.get(user_provided_user_id, conversation_id, skip_user_id_check) is None


def test_list_conversations(cache):
    """Test listing conversations for a user."""
    # Create multiple conversations
    conversation_id_1 = suid.get_suid()
    conversation_id_2 = suid.get_suid()

    cache.insert_or_append(constants.DEFAULT_USER_UID, conversation_id_1, cache_entry_1)
    cache.insert_or_append(constants.DEFAULT_USER_UID, conversation_id_2, cache_entry_2)

    conversations = cache.list(constants.DEFAULT_USER_UID)

    assert len(conversations) == 2
    assert conversation_id_1 in conversations
    assert conversation_id_2 in conversations


def test_list_conversations_skip_user_id_check(cache):
    """Test listing conversations for a user."""
    # Create multiple conversations
    conversation_id_1 = suid.get_suid()
    conversation_id_2 = suid.get_suid()
    skip_user_id_check = True

    cache.insert_or_append(
        user_provided_user_id, conversation_id_1, cache_entry_1, skip_user_id_check
    )
    cache.insert_or_append(
        user_provided_user_id, conversation_id_2, cache_entry_2, skip_user_id_check
    )

    conversations = cache.list(user_provided_user_id, skip_user_id_check)

    assert len(conversations) == 2
    assert conversation_id_1 in conversations
    assert conversation_id_2 in conversations


def test_list_no_conversations(cache):
    """Test listing conversations for a user with no conversations."""
    conversations = cache.list(constants.DEFAULT_USER_UID)
    assert len(conversations) == 0


def test_list_after_delete(cache):
    """Test listing conversations after deleting some."""
    conversation_id_1 = suid.get_suid()
    conversation_id_2 = suid.get_suid()

    cache.insert_or_append(constants.DEFAULT_USER_UID, conversation_id_1, cache_entry_1)
    cache.insert_or_append(constants.DEFAULT_USER_UID, conversation_id_2, cache_entry_2)

    cache.delete(constants.DEFAULT_USER_UID, conversation_id_1)

    conversations = cache.list(constants.DEFAULT_USER_UID)
    assert len(conversations) == 1
    assert conversation_id_2 in conversations
    assert conversation_id_1 not in conversations


def test_ready(cache):
    """Test if in memory cache always report ready."""
    assert cache.ready()


improper_user_uuids = [
    None,
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
def test_list_improper_user_id(cache, uuid):
    """Test list with invalid user ID."""
    with pytest.raises(ValueError, match=f"Invalid user ID {uuid}"):
        cache.list(uuid)


@pytest.mark.parametrize("uuid", improper_user_uuids)
def test_delete_improper_user_id(cache, uuid):
    """Test delete with invalid user ID."""
    with pytest.raises(ValueError, match=f"Invalid user ID {uuid}"):
        cache.delete(uuid, conversation_id)


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
    mc = InMemoryCacheConfig({"max_entries": "10"})
    cache1 = InMemoryCache(mc)
    cache2 = InMemoryCache(mc)
    assert cache1 is cache2
