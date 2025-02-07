"""Unit tests for RedisCache class."""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ols import constants
from ols.app.models.config import RedisConfig
from ols.app.models.models import CacheEntry
from ols.src.cache.redis_cache import RedisCache
from ols.utils import suid
from tests.mock_classes.mock_redis_client import MockRedisClient

conversation_id = suid.get_suid()
cache_entry_1 = CacheEntry(
    query=HumanMessage("user message1"), response=AIMessage("ai message1")
)
cache_entry_2 = CacheEntry(
    query=HumanMessage("user message2"), response=AIMessage("ai message2")
)
user_provided_user_id = "test-user1"


@pytest.fixture
def cache():
    """Fixture with constucted and initialized Redis cache object."""
    # we don't want to connect to real Redis from unit tests
    # with patch("ols.src.cache.redis_cache.RedisCache.initialize_redis"):
    with patch("redis.StrictRedis", new=MockRedisClient):
        return RedisCache(RedisConfig({}))


def test_insert_or_append(cache):
    """Test the behavior of insert_or_append method."""
    assert cache.get(constants.DEFAULT_USER_UID, conversation_id) is None

    cache.insert_or_append(
        constants.DEFAULT_USER_UID,
        conversation_id,
        cache_entry_1,
    )

    assert cache.get(constants.DEFAULT_USER_UID, conversation_id) == [cache_entry_1]


def test_insert_or_append_existing_key(cache):
    """Test the behavior of insert_or_append method for existing item."""
    # conversation IDs are separated by users
    # this UUID is different from DEFAULT_USER_UID
    user_uuid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert cache.get(user_uuid, conversation_id) is None

    cache.insert_or_append(user_uuid, conversation_id, cache_entry_1)
    cache.insert_or_append(user_uuid, conversation_id, cache_entry_2)

    assert cache.get(user_uuid, conversation_id) == [cache_entry_1, cache_entry_2]


def test_insert_or_append_skip_user_id_check(cache):
    """Test the behavior of insert_or_append method for existing item."""
    skip_user_id_check = True
    assert cache.get(user_provided_user_id, conversation_id, skip_user_id_check) is None

    cache.insert_or_append(
        user_provided_user_id, conversation_id, cache_entry_1, skip_user_id_check
    )
    cache.insert_or_append(
        user_provided_user_id, conversation_id, cache_entry_2, skip_user_id_check
    )

    assert cache.get(user_provided_user_id, conversation_id, skip_user_id_check) == [
        cache_entry_1,
        cache_entry_2,
    ]


def test_get_nonexistent_key(cache):
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
    user_id = suid.get_suid()
    conversations = cache.list(user_id)
    assert len(conversations) == 0


def test_list_after_delete(cache):
    """Test listing conversations after deleting some."""
    conversation_id_1 = suid.get_suid()
    conversation_id_2 = suid.get_suid()
    user_id = suid.get_suid()

    cache.insert_or_append(user_id, conversation_id_1, cache_entry_1)
    cache.insert_or_append(user_id, conversation_id_2, cache_entry_2)

    cache.delete(user_id, conversation_id_1)

    conversations = cache.list(user_id)
    assert len(conversations) == 1
    assert conversation_id_2 in conversations
    assert conversation_id_1 not in conversations


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
def test_get_improper_user_id(cache, uuid):
    """Test how improper user ID is handled."""
    with pytest.raises(ValueError, match=f"Invalid user ID {uuid}"):
        cache.get(uuid, conversation_id)


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


def test_get_improper_conversation_id(cache):
    """Test how improper conversation ID is handled."""
    with pytest.raises(ValueError, match="Invalid conversation ID"):
        cache.get(constants.DEFAULT_USER_UID, "this-is-not-valid-uuid")


def test_singleton_pattern():
    """Test if in memory cache exists as one instance in memory."""
    cache1 = RedisCache(RedisConfig({}))
    cache2 = RedisCache(RedisConfig({}))
    assert cache1 is cache2


def test_initialize_redis_no_password():
    """Test Redis initialization code when no password is specified."""
    with patch("redis.StrictRedis", new=MockRedisClient):
        config = RedisConfig({})
        cache = RedisCache(config)
        cache.initialize_redis(config)
        assert "password" not in cache.redis_client.kwargs


def test_initialize_redis_with_password():
    """Test Redis initialization code when password is specified."""
    with patch("redis.StrictRedis", new=MockRedisClient):
        config = RedisConfig(
            {
                "password_path": "tests/config/redis_password.txt",
            }
        )
        cache = RedisCache(config)
        cache.initialize_redis(config)
        assert cache.redis_client.kwargs["password"] == "redis_password"  # noqa: S105


def test_initialize_redis_with_no_ca_cert_path():
    """Test Redis initialization code when no CA certificate path is specified."""
    with patch("redis.StrictRedis", new=MockRedisClient):
        config = RedisConfig({})
        cache = RedisCache(config)
        cache.initialize_redis(config)
        assert "ssl" not in cache.redis_client.kwargs
        assert "ssl_cert_reqs" not in cache.redis_client.kwargs
        assert "ssl_ca_certs" not in cache.redis_client.kwargs


def test_initialize_redis_with_tls_certs():
    """Test Redis initialization code when CA certificate path is specified."""
    with patch("redis.StrictRedis", new=MockRedisClient):
        config = RedisConfig(
            {
                "ca_cert_path": "test/config/redis_ca_cert.crt",
            }
        )
        cache = RedisCache(config)
        cache.initialize_redis(config)
        assert cache.redis_client.kwargs["ssl"] is True
        assert cache.redis_client.kwargs["ssl_cert_reqs"] == "required"
        assert (
            cache.redis_client.kwargs["ssl_ca_certs"] == "test/config/redis_ca_cert.crt"
        )


def test_initialize_redis_default_retry_settings():
    """Test Redis initialization code, default retry settings."""
    with patch("redis.StrictRedis", new=MockRedisClient):
        config = RedisConfig({})
        cache = RedisCache(config)
        cache.initialize_redis(config)
        assert cache.redis_client.kwargs["retry_on_timeout"] is True
        assert cache.redis_client.kwargs["retry_on_error"] is not None
        assert cache.redis_client.kwargs["retry"]._retries == 3


def test_initialize_redis_retry_settings():
    """Test Redis initialization code when retry settings is specified."""
    with patch("redis.StrictRedis", new=MockRedisClient):
        config = RedisConfig(
            {
                "retry_on_error": "false",
                "retry_on_timeout": "false",
                "number_of_retries": 100,
            }
        )
        cache = RedisCache(config)
        cache.initialize_redis(config)
        assert cache.redis_client.kwargs["retry_on_timeout"] is False
        assert cache.redis_client.kwargs["retry_on_error"] is None
        assert cache.redis_client.kwargs["retry"]._retries == 100

    with patch("redis.StrictRedis", new=MockRedisClient):
        config = RedisConfig(
            {
                "retry_on_error": "False",
                "retry_on_timeout": "False",
                "number_of_retries": 100,
            }
        )
        cache = RedisCache(config)
        cache.initialize_redis(config)
        assert cache.redis_client.kwargs["retry_on_timeout"] is False
        assert cache.redis_client.kwargs["retry_on_error"] is None
        assert cache.redis_client.kwargs["retry"]._retries == 100

    with patch("redis.StrictRedis", new=MockRedisClient):
        config = RedisConfig(
            {
                "retry_on_error": "true",
                "retry_on_timeout": "true",
                "number_of_retries": 10,
            }
        )
        cache = RedisCache(config)
        cache.initialize_redis(config)
        assert cache.redis_client.kwargs["retry_on_timeout"] is True
        assert cache.redis_client.kwargs["retry_on_error"] is not None
        assert cache.redis_client.kwargs["retry"]._retries == 10

    with patch("redis.StrictRedis", new=MockRedisClient):
        config = RedisConfig(
            {
                "retry_on_error": "True",
                "retry_on_timeout": "True",
                "number_of_retries": 10,
            }
        )
        cache = RedisCache(config)
        cache.initialize_redis(config)
        assert cache.redis_client.kwargs["retry_on_timeout"] is True
        assert cache.redis_client.kwargs["retry_on_error"] is not None
        assert cache.redis_client.kwargs["retry"]._retries == 10
