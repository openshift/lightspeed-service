"""Cache factory class."""

from ols import constants
from ols.app.models.config import ConversationCacheConfig
from ols.src.cache.cache import Cache
from ols.src.cache.in_memory_cache import InMemoryCache
from ols.src.cache.postgres_cache import PostgresCache


class CacheFactory:
    """Cache factory class."""

    @staticmethod
    def conversation_cache(config: ConversationCacheConfig) -> Cache:
        """Create an instance of Cache based on loaded configuration.

        Returns:
            An instance of `Cache` (either `PostgresCache` or `InMemoryCache`).
        """
        match config.type:
            case constants.CACHE_TYPE_MEMORY:
                return InMemoryCache(config.memory)
            case constants.CACHE_TYPE_POSTGRES:
                return PostgresCache(config.postgres)
            case _:
                raise ValueError(
                    f"Invalid cache type: {config.type}. "
                    f"Use '{constants.CACHE_TYPE_POSTGRES}' or "
                    f"'{constants.CACHE_TYPE_MEMORY}' options."
                )
