"""In-memory LRU cache implemenetation."""

from __future__ import annotations

import threading
from collections import deque
from typing import Union

from ols.src.cache.cache import Cache


class InMemoryCache(Cache):
    """An in-memory LRU cache implementation in O(1) time."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls: type[InMemoryCache], size: int) -> InMemoryCache:
        """Implement Singleton pattern with thread safety."""
        with cls._lock:
            if not cls._instance:
                cls._instance = super(InMemoryCache, cls).__new__(cls)
                cls._instance.initialize_cache(size)
        return cls._instance

    def initialize_cache(self, size: int) -> None:
        """Initialize the InMemoryCache."""
        self.capacity = size
        self.deque: deque[str] = deque()
        self.cache: dict[str, str] = {}

    def get(self, user_id: str, conversation_id: str) -> Union[str, None]:
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

    def insert_or_append(self, user_id: str, conversation_id: str, value: str) -> None:
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
                self.cache[key] = value
            else:
                self.deque.remove(key)
                old_value = self.cache[key]
                self.cache[key] = old_value + "\n" + value
            self.deque.appendleft(key)
