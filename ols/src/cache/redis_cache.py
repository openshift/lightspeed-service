"""Cache that uses Redis to store cached values."""

import logging
import threading

import redis

from ols import constants
from ols.app.models.config import RedisConfig
from ols.src.cache.cache import Cache
from ols.src.cache.conversation import Conversation

logger = logging.getLogger(__name__)


class RedisMaxRetryError(Exception):
    """RedisMaxRetryError."""


# TODO
# Good for on-premise hosting for now
# Extend it to distributed setting using cloud offerings
class RedisCache(Cache):
    """Cache that uses Redis to store cached values."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls: type["RedisCache"], config: RedisConfig) -> "RedisCache":
        """Create a new instance of the `RedisCache` class."""
        with cls._lock:
            if not cls._instance:
                cls._instance = super(RedisCache, cls).__new__(cls)
                cls._instance.initialize_redis(config)
        return cls._instance

    def initialize_redis(self, config: RedisConfig) -> None:
        """Initialize the Redis client and logger.

        This method sets up the Redis client with custom configuration parameters.
        """
        kwargs = {}
        if config.credentials is not None:
            if config.credentials.username is not None:
                kwargs["username"] = config.credentials.username
            if config.credentials.password is not None:
                kwargs["password"] = config.credentials.password
        self.redis_client = redis.StrictRedis(
            host=config.host,
            port=config.port,
            decode_responses=True,
            **kwargs,
        )
        # Set custom configuration parameters
        self.redis_client.config_set("maxmemory", config.max_memory)
        self.redis_client.config_set("maxmemory-policy", config.max_memory_policy)

    def get(self, user_id: str, conversation_id: str) -> list[Conversation] | None:
        """Get the value associated with the given key.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.

        Returns:
            The value associated with the key, or None if not found.
        """
        key = super().construct_key(user_id, conversation_id)
        value = self.redis_client.get(key)
        # GET operation might return Awaitable value .
        return value if value else None

    def insert_or_append(
        self, user_id: str, conversation_id: str, value: Conversation
    ) -> None:
        """Set the value associated with the given key.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            value: The value to set.

        Raises:
            RedisConnectionError: If item is unable to update after REDIS_MAX_RETRY.
        """
        key = super().construct_key(user_id, conversation_id)
        old_value = self.get(user_id, conversation_id)
        retry_count = 0
        while retry_count < constants.REDIS_MAX_RETRY:
            logger.debug("Updating redis cache ")
            with self._lock:
                if old_value:
                    old_value.append(value)
                    try:
                        self.redis_client.set(key, old_value)
                        break
                    except Exception as redis_err:
                        logger.error(
                            f"Exception during updating the cache: {redis_err}"
                        )
                        retry_count += 1
                else:
                    values = [value]
                    try:
                        self.redis_client.set(key, values)
                        break
                    except Exception as redis_err:
                        logger.error(
                            f"Exception during creating the cache: {redis_err}"
                        )
                        retry_count += 1

        raise RedisMaxRetryError("Updating cache failed after max retries")
