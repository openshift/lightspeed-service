# cache_factory.py


from app.models.config import ConversationCacheConfig
from src import constants
from src.cache.cache import Cache
from src.cache.in_memory_cache import InMemoryCache
from src.cache.redis_cache import RedisCache


class CacheFactory:
    @staticmethod
    def conversation_cache(config: ConversationCacheConfig) -> Cache:
        """Factory method to create an instance of Cache based on environment variable.

        Returns:
            Cache: An instance of Cache (either RedisCache or InMemoryCache).
        """

        if config.type == constants.REDIS_CACHE:
            return RedisCache()
        elif config.type == constants.IN_MEMORY_CACHE:
            return InMemoryCache(config.memory.max_entries)
        else:
            raise ValueError(
                f"Invalid cache type: {config.type}. Use 'redis' or 'in-memory' options."
            )
