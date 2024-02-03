"""In-memory LRU cache implemenetation."""

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ols.app.models.config import MemoryConfig
    from ols.src.cache.conversation import Conversation

from ols.src.cache.cache import Cache


class InMemoryCache(Cache):
    """An in-memory LRU cache implementation in O(1) time."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls: type[InMemoryCache], config: MemoryConfig) -> InMemoryCache:
        """Implement Singleton pattern with thread safety."""
        with cls._lock:
            if not cls._instance:
                cls._instance = super(InMemoryCache, cls).__new__(cls)
                cls._instance.initialize_cache(config)
        return cls._instance

    def initialize_cache(self, config: MemoryConfig) -> None:
        """Initialize the InMemoryCache."""
        self.capacity = config.max_entries
        self.deque: deque[str] = deque()
        self.cache: dict[str, list[Conversation]] = {}

    def get(self, user_id: str, conversation_id: str) -> list[Conversation] | None:
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
        return self.cache[key]

    def insert_or_append(
        self, user_id: str, conversation_id: str, value: Conversation
    ) -> None:
        """Set the value if a key is not present or else simply appends.

        Args:
          user_id: User identification.
          conversation_id: Conversation ID unique for given user.
          value: The value to associate with the key.
        """
        key = super().construct_key(user_id, conversation_id)

        with self._lock:
            if key not in self.cache:
                if len(self.deque) == self.capacity:
                    oldest = self.deque.pop()
                    del self.cache[oldest]
                values = [value]
                self.cache[key] = values
            else:
                self.deque.remove(key)
                self.cache[key].append(value)
            self.deque.appendleft(key)
