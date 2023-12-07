# cache_factory.py

import os
from abc import ABC, abstractmethod
from typing import Union
from lightspeed_service import constants
from lightspeed_service.cache.in_memory_cache import InMemoryCache
from lightspeed_service.cache.redis_cache import RedisCache


class Cache(ABC):
    @abstractmethod
    def get(self, key: str) -> Union[str, None]:
        """Abstract method to retrieve a value from the cache.

        Args:
            key (str): The key associated with the value.

        Returns:
            Union[str, None]: The value associated with the key, or None if not found.
        """
        pass

    @abstractmethod
    def insert_or_append(self, key: str, value: str) -> None:
        """Abstract method to store a value in the cache.

        Args:
            key (str): The key to associate with the value.
            value (str): The value to be stored in the cache.

        Returns:
            None
        """
        pass


class CacheFactory:
    @staticmethod
    def conversation_cache() -> Cache:
        """Factory method to create an instance of Cache based on environment variable.

        Returns:
            Cache: An instance of Cache (either RedisCache or InMemoryCache).
        """
        cache_type = os.environ.get(
            "OLS_CONVERSATION_CACHE", constants.IN_MEMORY_CACHE
        ).lower()

        if cache_type == constants.REDIS_CACHE:
            return RedisCache()
        elif cache_type == constants.IN_MEMORY_CACHE:
            return InMemoryCache()
        else:
            raise ValueError("Invalid cache type. Use 'redis' or 'in-memory' options.")
