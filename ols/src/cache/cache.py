"""Abstract class that is parent for all cache implementations."""

from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.messages.base import BaseMessage

from ols.utils.suid import check_suid


class Cache(ABC):
    """Abstract class that is parent for all cache implementations.

    Cache entries are identified by compound key that consists of
    user ID and conversation ID. Application logic must ensure that
    user will be able to store and retrieve values that have the
    correct user ID only. This means that user won't be able to
    read or modify other users conversations.
    """

    # separator between parts of compond key
    COMPOUND_KEY_SEPARATOR = ":"

    @staticmethod
    def _check_user_id(user_id: str) -> None:
        """Check if given user ID is valid."""
        if not check_suid(user_id):
            raise ValueError(f"Invalid user ID {user_id}")

    @staticmethod
    def _check_conversation_id(conversation_id: str) -> None:
        """Check if given conversation ID is a valid UUID (including optional dashes)."""
        if not check_suid(conversation_id):
            raise ValueError(f"Invalid conversation ID {conversation_id}")

    @staticmethod
    def construct_key(user_id: str, conversation_id: str) -> str:
        """Construct key to cache."""
        Cache._check_user_id(user_id)
        Cache._check_conversation_id(conversation_id)
        return f"{user_id}{Cache.COMPOUND_KEY_SEPARATOR}{conversation_id}"

    @abstractmethod
    def get(self, user_id: str, conversation_id: str) -> Optional[list[BaseMessage]]:
        """Abstract method to retrieve a value from the cache.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.

        Returns:
            The value associated with the key, or None if not found.
        """

    @abstractmethod
    def insert_or_append(
        self, user_id: str, conversation_id: str, value: list[BaseMessage]
    ) -> None:
        """Abstract method to store a value in the cache.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            value: The value to be stored in the cache.
        """
