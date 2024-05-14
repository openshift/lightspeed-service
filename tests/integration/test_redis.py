"""Integration tests for real Redis behaviour."""

import pytest

from ols.app.endpoints.ols import ai_msg, human_msg
from ols.app.models.config import RedisConfig
from ols.src.cache.redis_cache import RedisCache

user_id = "00000000-0000-0000-0000-000000000001"
conversation_id = "00000000-0000-0000-0000-000000000002"


@pytest.mark.redis()
def setup():
    """Setups the Redis client."""
    global redis_cache

    # please note that the setup expect Redis running locally on default port
    redis_config = RedisConfig(
        {
            "host": "localhost",
            "port": 6379,
            "max_memory": "100mb",
            "max_memory_policy": "allkeys-lru",
            "retry_on_error": "false",
            "retry_on_timeout": "false",
        }
    )

    redis_cache = RedisCache(redis_config)


@pytest.mark.redis()
def test_conversation_in_redis():
    """Check the elementary GET operation and insert_or_append operation."""
    # make sure the cache is empty
    redis_cache.redis_client.delete(user_id + ":" + conversation_id)

    # the initial value should be empty
    retrieved = redis_cache.get(user_id, conversation_id)
    assert retrieved is None

    # insert some conversation
    conversation = [
        human_msg("First human message"),
        ai_msg("First AI response"),
    ]
    redis_cache.insert_or_append(user_id, conversation_id, conversation)

    # check what is stored in conversation cache
    retrieved = redis_cache.get(user_id, conversation_id)
    assert retrieved is not None

    # just the initial conversation should be stored
    assert retrieved == conversation

    # append more conversation
    conversation2 = [
        human_msg("Second human message"),
        ai_msg("Second AI response"),
    ]
    redis_cache.insert_or_append(user_id, conversation_id, conversation2)

    # check what is stored in conversation cache
    retrieved = redis_cache.get(user_id, conversation_id)
    assert retrieved is not None

    # now both conversations should be stored
    assert retrieved == conversation + conversation2
