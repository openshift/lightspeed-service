"""Cache that uses Postgres to store cached values."""

from typing import Optional

from langchain_core.messages.base import BaseMessage

from ols.app.models.config import PostgresConfig
from ols.src.cache.cache import Cache


class PostgresCache(Cache):
    """Cache that uses Postgres to store cached values."""

    def __init__(self, config: PostgresConfig):
        """Create a new instance of Postgres cache."""

    def get(self, user_id: str, conversation_id: str) -> Optional[list[BaseMessage]]:
        """Get the value associated with the given key.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.

        Returns:
            The value associated with the key, or None if not found.
        """
        return None

    def insert_or_append(
        self, user_id: str, conversation_id: str, value: list[BaseMessage]
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
