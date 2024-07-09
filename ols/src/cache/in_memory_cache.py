"""In-memory LRU cache implemenetation."""

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING, Any

from ols.app.models.models import CacheEntry

if TYPE_CHECKING:
    from ols.app.models.config import InMemoryCacheConfig
from ols.src.cache.cache import Cache


class InMemoryCache(Cache):
    """An in-memory LRU cache implementation in O(1) time."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls: type[InMemoryCache], config: InMemoryCacheConfig) -> InMemoryCache:
        """Implement Singleton pattern with thread safety."""
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance.initialize_cache(config)
        return cls._instance

    def initialize_cache(self, config: InMemoryCacheConfig) -> None:
        """Initialize the InMemoryCache."""
        self.capacity = config.max_entries
        self.deque: deque[str] = deque()
        self.cache: dict[str, list[dict[str, Any]]] = {}

    def get(self, user_id: str, conversation_id: str) -> list[CacheEntry]:
        """Get the value associated with the given key.

        Args:
          user_id: User identification.
          conversation_id: Conversation ID unique for given user.

        Returns:
          The value associated with the key, or `None` if the key is not present.
        """
        key = super().construct_key(user_id, conversation_id)

        if key not in self.cache:
            return None

        self.deque.remove(key)
        self.deque.appendleft(key)
        value = self.cache[key].copy()
        cache_entry = [CacheEntry.from_dict(cache_entry) for cache_entry in value]
        return cache_entry

    def insert_or_append(
        self,
        user_id: str,
        conversation_id: str,
        cache_entry: CacheEntry,
    ) -> None:
        """Set the value if a key is not present or else simply appends.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            cache_entry: The `CacheEntry` object to store.
        """
        key = super().construct_key(user_id, conversation_id)
        value = cache_entry.to_dict()

        with self._lock:
            if key not in self.cache:
                if len(self.deque) == self.capacity:
                    oldest = self.deque.pop()
                    del self.cache[oldest]
                self.cache[key] = [value]
            else:
                self.deque.remove(key)
                old_value = self.cache[key]
                old_value.append(value)
                self.cache[key] = old_value
            self.deque.appendleft(key)
