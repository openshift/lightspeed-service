import os
import redis
from typing import Union
import threading
import src.constants as constants
from dotenv import load_dotenv
from src.cache.cache import Cache


load_dotenv()


# TODO
# Good for on-premise hosting for now
# Extend it to distributed setting using cloud offerings
class RedisCache():
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """
        Create a new instance of the RedisCache class.

        Returns:
            RedisCache: The singleton instance of RedisCache.
        """
        with cls._lock:
            if not cls._instance:
                cls._instance = super(RedisCache, cls).__new__(cls)
                cls._instance.initialize_redis()
        return cls._instance

    def initialize_redis(self):
        """
        Initialize the Redis client and logger.

        This method sets up the Redis client with custom configuration parameters.

        Returns:
            None
        """
        self.redis_client = redis.StrictRedis(
            host=os.environ.get("REDIS_CACHE_HOST", constants.REDIS_CACHE_HOST),
            port=os.environ.get("REDIS_CACHE_PORT", constants.REDIS_CACHE_PORT),
            decode_responses=True,
        )
        # Set custom configuration parameters
        self.redis_client.config_set("maxmemory", constants.REDIS_CACHE_MAX_MEMORY)
        self.redis_client.config_set(
            "maxmemory-policy", constants.REDIS_CACHE_MAX_MEMORY_POLICY
        )

    def get(self, key: str) -> Union[str, None]:
        """
        Get the value associated with the given key.

        Args:
            key (str): The key for the desired value.

        Returns:
            Union[str, None]: The value associated with the key, or None if not found.
        """
        return self.redis_client.get(key)

    def insert_or_append(self, key: str, value: str) -> None:
        """
        Set the value associated with the given key.

        Args:
            key (str): The key for the value.
            value (str): The value to set.

        Returns:
            None
        """
        oldValue = self.get(key)
        with self._lock:
            if oldValue:
                self.redis_client.set(key, oldValue + "\n" + value)
            else:
                self.redis_client.set(key, value)
