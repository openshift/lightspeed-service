# cache_factory.py

import os
from src import constants
from src.cache.cache import Cache
from src.cache.in_memory_cache import InMemoryCache
from src.cache.redis_cache import RedisCache


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
