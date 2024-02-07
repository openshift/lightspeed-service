"""Cache that uses Redis to store cached values."""

import threading
from typing import Optional

import redis

from ols.app.models.config import RedisConfig
from ols.src.cache.cache import Cache


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

    def get(self, user_id: str, conversation_id: str) -> Optional[str]:
        """Get the value associated with the given key.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.

        Returns:
            The value associated with the key, or None if not found.
        """
        key = super().construct_key(user_id, conversation_id)

        return self.redis_client.get(key)

    def insert_or_append(self, user_id: str, conversation_id: str, value: str) -> None:
        """Set the value associated with the given key.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            value: The value to set.

        Raises:
            OutOfMemoryError: If item is evicted when Redis allocated
                memory is higher than maxmemory.
        """
        key = super().construct_key(user_id, conversation_id)

        old_value = self.get(user_id, conversation_id)
        with self._lock:
            if old_value:
                self.redis_client.set(key, old_value + "\n" + value)
            else:
                self.redis_client.set(key, value)
