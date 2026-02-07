"""Unit tests for InMemoryCache class."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ols import constants
from ols.app.models.config import InMemoryCacheConfig
from ols.app.models.models import CacheEntry
from ols.src.cache.in_memory_cache import InMemoryCache
from ols.utils import suid

CONVERSATION_ID = suid.get_suid()
USER_PROVIDED_USER_ID = "test-user1"
CACHE_ENTRY_1 = CacheEntry(
    query=HumanMessage("user message1"), response=AIMessage("ai message1")
)
CACHE_ENTRY_2 = CacheEntry(
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
        CONVERSATION_ID,
        CACHE_ENTRY_1,
    )

    history, was_limited = cache.get(constants.DEFAULT_USER_UID, CONVERSATION_ID)
    assert history == [CACHE_ENTRY_1]
    assert was_limited is False


def test_insert_or_append_skip_user_id_check(cache):
    """Test the behavior of insert_or_append method."""
    skip_user_id_check = True
    cache.insert_or_append(
        USER_PROVIDED_USER_ID, CONVERSATION_ID, CACHE_ENTRY_1, skip_user_id_check
    )

    history, was_limited = cache.get(
        USER_PROVIDED_USER_ID, CONVERSATION_ID, skip_user_id_check
    )
    assert history == [CACHE_ENTRY_1]
    assert was_limited is False


def test_insert_or_append_existing_key(cache):
    """Test the behavior of insert_or_append method for existing item."""
    cache.insert_or_append(
        constants.DEFAULT_USER_UID,
        CONVERSATION_ID,
        CACHE_ENTRY_1,
    )
    cache.insert_or_append(
        constants.DEFAULT_USER_UID,
        CONVERSATION_ID,
        CACHE_ENTRY_2,
    )
    expected_cache = [
        CACHE_ENTRY_1,
        CACHE_ENTRY_2,
    ]

    history, was_limited = cache.get(constants.DEFAULT_USER_UID, CONVERSATION_ID)
    assert history == expected_cache
    assert was_limited is False


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
            CONVERSATION_ID,
            value,
        )

    # Ensure the oldest entry is evicted
    history, was_limited = cache.get(f"{user_name_prefix}0", CONVERSATION_ID)
    assert history == []
    assert was_limited is False
    # Ensure the newest entry is still present
    expected_result = [CacheEntry(query=HumanMessage(f"user query {i}"))]
    history, was_limited = cache.get(f"{user_name_prefix}{capacity}", CONVERSATION_ID)
    assert history == expected_result
    assert was_limited is False


def test_insert_or_append_eviction(cache):
    """Test if items in cache eviction with defined capacity is handled correctly."""
    # remove last hex digit from user UUID
    user_name_prefix = constants.DEFAULT_USER_UID[:-1]

    capacity = 5
    cache.capacity = capacity
    cache.insert_or_append(
        f"{user_name_prefix}{0}",
        CONVERSATION_ID,
        CacheEntry(query=HumanMessage("user query 0 intial entry")),
    )
    for i in range(capacity):
        user = f"{user_name_prefix}{i}"
        value = CacheEntry(query=HumanMessage(f"user query {i}"))
        cache.insert_or_append(
            user,
            CONVERSATION_ID,
            value,
        )

    # Ensure the oldest entry is evicted
    expected_result = [CacheEntry(query=HumanMessage("user query 0"))]
    history, was_limited = cache.get(f"{user_name_prefix}0", CONVERSATION_ID)
    assert history == expected_result
    assert was_limited is False
    # Ensure the newest entry is still present
    expected_result = [CacheEntry(query=HumanMessage(f"user query {i}"))]
    history, was_limited = cache.get(
        f"{user_name_prefix}{capacity - 1}", CONVERSATION_ID
    )
    assert history == expected_result
    assert was_limited is False


def test_get_nonexistent_user(cache):
    """Test how non-existent items are handled by the cache."""
    # this UUID is different from DEFAULT_USER_UID
    history, was_limited = cache.get(
        "ffffffff-ffff-ffff-ffff-ffffffffffff", CONVERSATION_ID
    )
    assert history == []
    assert was_limited is False


def test_delete_existing_conversation(cache):
    """Test deleting an existing conversation."""
    cache.insert_or_append(constants.DEFAULT_USER_UID, CONVERSATION_ID, CACHE_ENTRY_1)

    result = cache.delete(constants.DEFAULT_USER_UID, CONVERSATION_ID)

    assert result is True
    history, was_limited = cache.get(constants.DEFAULT_USER_UID, CONVERSATION_ID)
    assert history == []
    assert was_limited is False


def test_delete_nonexistent_conversation(cache):
    """Test deleting a conversation that doesn't exist."""
    result = cache.delete(constants.DEFAULT_USER_UID, CONVERSATION_ID)
    assert result is False


def test_delete_improper_conversation_id(cache):
    """Test delete with invalid conversation ID."""
    with pytest.raises(ValueError, match="Invalid conversation ID"):
        cache.delete(constants.DEFAULT_USER_UID, "invalid-id")


def test_delete_skip_user_id_check(cache):
    """Test deleting an existing conversation."""
    skip_user_id_check = True
    cache.insert_or_append(
        USER_PROVIDED_USER_ID, CONVERSATION_ID, CACHE_ENTRY_1, skip_user_id_check
    )

    result = cache.delete(USER_PROVIDED_USER_ID, CONVERSATION_ID, skip_user_id_check)

    assert result is True
    history, was_limited = cache.get(
        USER_PROVIDED_USER_ID, CONVERSATION_ID, skip_user_id_check
    )
    assert history == []
    assert was_limited is False


def test_list_conversations(cache):
    """Test listing conversations for a user."""
    # Create multiple conversations
    conversation_id_1 = suid.get_suid()
    conversation_id_2 = suid.get_suid()

    cache.insert_or_append(constants.DEFAULT_USER_UID, conversation_id_1, CACHE_ENTRY_1)
    cache.insert_or_append(constants.DEFAULT_USER_UID, conversation_id_2, CACHE_ENTRY_2)

    conversations = cache.list(constants.DEFAULT_USER_UID)

    assert len(conversations) == 2
    conversation_ids = [c.conversation_id for c in conversations]
    assert conversation_id_1 in conversation_ids
    assert conversation_id_2 in conversation_ids


def test_list_conversations_skip_user_id_check(cache):
    """Test listing conversations for a user."""
    # Create multiple conversations
    conversation_id_1 = suid.get_suid()
    conversation_id_2 = suid.get_suid()
    skip_user_id_check = True

    cache.insert_or_append(
        USER_PROVIDED_USER_ID, conversation_id_1, CACHE_ENTRY_1, skip_user_id_check
    )
    cache.insert_or_append(
        USER_PROVIDED_USER_ID, conversation_id_2, CACHE_ENTRY_2, skip_user_id_check
    )

    conversations = cache.list(USER_PROVIDED_USER_ID, skip_user_id_check)

    assert len(conversations) == 2
    conversation_ids = [c.conversation_id for c in conversations]
    assert conversation_id_1 in conversation_ids
    assert conversation_id_2 in conversation_ids


def test_list_no_conversations(cache):
    """Test listing conversations for a user with no conversations."""
    conversations = cache.list(constants.DEFAULT_USER_UID)
    assert len(conversations) == 0


def test_list_after_delete(cache):
    """Test listing conversations after deleting some."""
    conversation_id_1 = suid.get_suid()
    conversation_id_2 = suid.get_suid()

    cache.insert_or_append(constants.DEFAULT_USER_UID, conversation_id_1, CACHE_ENTRY_1)
    cache.insert_or_append(constants.DEFAULT_USER_UID, conversation_id_2, CACHE_ENTRY_2)

    cache.delete(constants.DEFAULT_USER_UID, conversation_id_1)

    conversations = cache.list(constants.DEFAULT_USER_UID)
    assert len(conversations) == 1
    conversation_ids = [c.conversation_id for c in conversations]
    assert conversation_id_2 in conversation_ids
    assert conversation_id_1 not in conversation_ids


def test_list_conversations_metadata(cache):
    """Test that list returns ConversationData with correct metadata."""
    conv_id = suid.get_suid()

    cache.insert_or_append(constants.DEFAULT_USER_UID, conv_id, CACHE_ENTRY_1)
    cache.insert_or_append(constants.DEFAULT_USER_UID, conv_id, CACHE_ENTRY_2)

    conversations = cache.list(constants.DEFAULT_USER_UID)

    assert len(conversations) == 1
    conv_data = conversations[0]
    assert conv_data.conversation_id == conv_id
    assert conv_data.message_count == 2
    assert conv_data.topic_summary == ""
    assert conv_data.last_message_timestamp > 0


def test_set_topic_summary(cache):
    """Test setting topic summary for a conversation."""
    conv_id = suid.get_suid()

    cache.insert_or_append(constants.DEFAULT_USER_UID, conv_id, CACHE_ENTRY_1)
    cache.set_topic_summary(constants.DEFAULT_USER_UID, conv_id, "Test Topic")

    conversations = cache.list(constants.DEFAULT_USER_UID)

    assert len(conversations) == 1
    assert conversations[0].topic_summary == "Test Topic"


def test_set_topic_summary_creates_metadata(cache):
    """Test that set_topic_summary creates metadata even if no cache entry exists."""
    conv_id = suid.get_suid()

    cache.set_topic_summary(constants.DEFAULT_USER_UID, conv_id, "New Topic")

    conversations = cache.list(constants.DEFAULT_USER_UID)

    assert len(conversations) == 1
    assert conversations[0].conversation_id == conv_id
    assert conversations[0].topic_summary == "New Topic"
    assert conversations[0].message_count == 0


def test_set_topic_summary_skip_user_id_check(cache):
    """Test setting topic summary with skip_user_id_check."""
    conv_id = suid.get_suid()
    skip_user_id_check = True

    cache.insert_or_append(
        USER_PROVIDED_USER_ID, conv_id, CACHE_ENTRY_1, skip_user_id_check
    )
    cache.set_topic_summary(
        USER_PROVIDED_USER_ID, conv_id, "User Topic", skip_user_id_check
    )

    conversations = cache.list(USER_PROVIDED_USER_ID, skip_user_id_check)

    assert len(conversations) == 1
    assert conversations[0].topic_summary == "User Topic"


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
        cache.delete(uuid, CONVERSATION_ID)


@pytest.mark.parametrize("uuid", improper_user_uuids)
def test_get_improper_user_id(cache, uuid):
    """Test how improper user ID is handled."""
    with pytest.raises(ValueError, match=f"Invalid user ID {uuid}"):
        cache.get(uuid, CONVERSATION_ID)


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


def test_get_with_limit(cache):
    """Test get method with limit parameter."""
    # Insert 5 entries
    for i in range(5):
        entry = CacheEntry(
            query=HumanMessage(f"query {i}"), response=AIMessage(f"response {i}")
        )
        cache.insert_or_append(constants.DEFAULT_USER_UID, CONVERSATION_ID, entry)

    # Get with limit less than total
    history, was_limited = cache.get(
        constants.DEFAULT_USER_UID, CONVERSATION_ID, limit=3
    )
    assert len(history) == 3
    assert was_limited is True
    # Should return last 3 messages
    assert history[0].query.content == "query 2"
    assert history[2].query.content == "query 4"

    # Get with limit equal to total
    history, was_limited = cache.get(
        constants.DEFAULT_USER_UID, CONVERSATION_ID, limit=5
    )
    assert len(history) == 5
    assert was_limited is False

    # Get with limit greater than total
    history, was_limited = cache.get(
        constants.DEFAULT_USER_UID, CONVERSATION_ID, limit=10
    )
    assert len(history) == 5
    assert was_limited is False

    # Get without limit
    history, was_limited = cache.get(constants.DEFAULT_USER_UID, CONVERSATION_ID)
    assert len(history) == 5
    assert was_limited is False
