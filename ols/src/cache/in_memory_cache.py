"""In-memory LRU cache implemenetation."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from ols.app.models.models import CacheEntry, ConversationData

if TYPE_CHECKING:
    from ols.app.models.config import InMemoryCacheConfig
# pylint: disable-next=C0413
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
        # pylint: disable=W0201
        self.capacity: int = int(config.max_entries)
        self.total_entries: int = 0
        self.deque: deque[str] = deque()
        self.cache: dict[str, list[dict[str, Any]]] = {}
        # Conversations metadata storage
        self._conversations: dict[str, ConversationData] = {}

    def get(
        self, user_id: str, conversation_id: str, skip_user_id_check: bool = False
    ) -> list[CacheEntry]:
        """Get the value associated with the given key.

        Args:
          user_id: User identification.
          conversation_id: Conversation ID unique for given user.
          skip_user_id_check: Skip user_id suid check.

        Returns:
          The value associated with the key, or `None` if the key is not present.
        """
        key = super().construct_key(user_id, conversation_id, skip_user_id_check)

        if key not in self.cache:
            return None

        self.deque.remove(key)
        self.deque.appendleft(key)
        value = self.cache[key].copy()
        return [CacheEntry.from_dict(cache_entry) for cache_entry in value]

    def insert_or_append(
        self,
        user_id: str,
        conversation_id: str,
        cache_entry: CacheEntry,
        skip_user_id_check: bool = False,
    ) -> None:
        """Set the value if a key is not present or else simply appends.

        Eviction policy:
          - Capacity is treated as number of message entries across all conversations.
          - When inserting causes total entries to exceed capacity, evict the oldest
            message(s) from the least-recently-used conversation(s) (tail of deque)
            until total_entries <= capacity.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            cache_entry: The `CacheEntry` object to store.
            skip_user_id_check: Skip user_id suid check.
        """
        key = super().construct_key(user_id, conversation_id, skip_user_id_check)
        value = cache_entry.to_dict()

        with self._lock:
            if key not in self.cache:
                self.cache[key] = [value]
            else:
                self.deque.remove(key)
                self.cache[key].append(value)
            self.deque.appendleft(key)
            self.total_entries += 1

            # Update conversations metadata
            current_time = time.time()
            if key in self._conversations:
                conv_data = self._conversations[key]
                self._conversations[key] = ConversationData(
                    conversation_id=conversation_id,
                    topic_summary=conv_data.topic_summary,
                    last_message_timestamp=current_time,
                    message_count=conv_data.message_count + 1,
                )
            else:
                self._conversations[key] = ConversationData(
                    conversation_id=conversation_id,
                    topic_summary="",
                    last_message_timestamp=current_time,
                    message_count=1,
                )

            # Evict oldest messages until we're within capacity
            if self.total_entries > self.capacity and self.deque:
                oldest_key = self.deque[-1]
                oldest_list = self.cache.get(oldest_key, [])
                del oldest_list[0]
                self.total_entries -= 1

                if len(oldest_list) == 0:
                    del self.cache[oldest_key]
                    self.deque.pop()
                    # Also remove from conversations metadata
                    if oldest_key in self._conversations:
                        del self._conversations[oldest_key]
                else:
                    self.cache[oldest_key] = oldest_list

    def delete(
        self, user_id: str, conversation_id: str, skip_user_id_check: bool = False
    ) -> bool:
        """Delete all entries for a given conversation.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            skip_user_id_check: Skip user_id suid check.

        Returns:
            bool: True if entries were deleted, False if key wasn't found.
        """
        key = super().construct_key(user_id, conversation_id, skip_user_id_check)

        with self._lock:
            if key not in self.cache:
                return False

            # Remove from both cache and deque
            self.total_entries -= len(self.cache[key])
            del self.cache[key]
            self.deque.remove(key)
            # Also remove from conversations metadata
            if key in self._conversations:
                del self._conversations[key]
            return True

    def list(
        self, user_id: str, skip_user_id_check: bool = False
    ) -> list[ConversationData]:
        """List all conversations for a given user_id.

        Args:
            user_id: User identification.
            skip_user_id_check: Skip user_id suid check.

        Returns:
            A list of ConversationData objects containing conversation_id,
            topic_summary, last_message_timestamp, and message_count.
        """
        conversations: list[ConversationData] = []
        super()._check_user_id(user_id, skip_user_id_check)
        prefix = f"{user_id}{Cache.COMPOUND_KEY_SEPARATOR}"

        with self._lock:
            conversations.extend(
                conv
                for key, conv in self._conversations.items()
                if key.startswith(prefix)
            )

        # Sort by last_message_timestamp descending
        conversations.sort(key=lambda x: x.last_message_timestamp, reverse=True)
        return conversations

    def set_topic_summary(
        self,
        user_id: str,
        conversation_id: str,
        topic_summary: str,
        skip_user_id_check: bool = False,
    ) -> None:
        """Set or update the topic summary for a conversation.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            topic_summary: The topic summary to store.
            skip_user_id_check: Skip user_id suid check.
        """
        key = super().construct_key(user_id, conversation_id, skip_user_id_check)

        with self._lock:
            current_time = time.time()
            if key in self._conversations:
                conv_data = self._conversations[key]
                self._conversations[key] = ConversationData(
                    conversation_id=conversation_id,
                    topic_summary=topic_summary,
                    last_message_timestamp=current_time,
                    message_count=conv_data.message_count,
                )
            else:
                self._conversations[key] = ConversationData(
                    conversation_id=conversation_id,
                    topic_summary=topic_summary,
                    last_message_timestamp=current_time,
                    message_count=0,
                )

    def ready(self) -> bool:
        """Check if the cache is ready.

           In memory cache is always ready.

        Returns:
            True if the cache is ready, False otherwise.
        """
        return True
