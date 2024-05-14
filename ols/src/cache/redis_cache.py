"""Cache that uses Redis to store cached values."""

import pickle
import threading
from typing import Any, Optional

import redis
from redis.backoff import ExponentialBackoff
from redis.exceptions import (
    BusyLoadingError,
    ConnectionError,
    RedisError,
)
from redis.retry import Retry

from ols.app.models.config import RedisConfig
from ols.src.cache.cache import Cache


class RedisCache(Cache):
    """Cache that uses Redis to store cached values."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls: type["RedisCache"], config: RedisConfig) -> "RedisCache":
        """Create a new instance of the `RedisCache` class."""
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance.initialize_redis(config)
        return cls._instance

    def initialize_redis(self, config: RedisConfig) -> None:
        """Initialize the Redis client and logger.

        This method sets up the Redis client with custom configuration parameters.
        """
        kwargs: dict[str, Any] = {}
        if config.password is not None:
            kwargs["password"] = config.password
        if config.ca_cert_path is not None:
            kwargs["ssl"] = True
            kwargs["ssl_cert_reqs"] = "required"
            kwargs["ssl_ca_certs"] = config.ca_cert_path

        # setup Redis retry logic
        retry: Optional[Retry] = None
        if config.number_of_retries is not None and config.number_of_retries > 0:
            retry = Retry(ExponentialBackoff(), config.number_of_retries)

        retry_on_error: Optional[list[type[RedisError]]] = None
        if config.retry_on_error:
            retry_on_error = [BusyLoadingError, ConnectionError]

        # initialize Redis client
        self.redis_client = redis.StrictRedis(
            host=str(config.host),
            port=int(config.port),
            decode_responses=False,  # we store serialized messages as bytes, not strings
            retry=retry,
            retry_on_timeout=bool(config.retry_on_timeout),
            retry_on_error=retry_on_error,
            **kwargs,
        )
        # Set custom configuration parameters
        self.redis_client.config_set("maxmemory", config.max_memory)
        self.redis_client.config_set("maxmemory-policy", config.max_memory_policy)

    def get(self, user_id: str, conversation_id: str) -> Optional[list[dict]]:
        """Get the value associated with the given key.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.

        Returns:
            The value associated with the key, or None if not found.
        """
        key = super().construct_key(user_id, conversation_id)

        value = self.redis_client.get(key)
        if value is None:
            return None
        return pickle.loads(value, errors="strict")  # noqa S301

    def insert_or_append(
        self, user_id: str, conversation_id: str, value: list[dict]
    ) -> None:
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

        with self._lock:
            old_value = self.get(user_id, conversation_id)
            if old_value:
                old_value.extend(value)
                self.redis_client.set(
                    key, pickle.dumps(old_value, protocol=pickle.HIGHEST_PROTOCOL)
                )
            else:
                self.redis_client.set(
                    key, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                )
