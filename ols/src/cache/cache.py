"""Abstract class that is parent for all cache implementations."""

import re
from abc import ABC, abstractmethod
from typing import Union


class Cache(ABC):
    """Abstract class that is parent for all cache implementations."""

    @staticmethod
    def _check_user_id(user_id: str):
        """Check if given user ID is valid."""
        # TODO: needs to be updated when we know the format
        if "/" in user_id:
            raise ValueError("Incorrect user ID {user_id}")

    @staticmethod
    def _check_conversation_id(conversation_id: str):
        """Check if given conversation ID is a valid UUID (including optional dashes)."""
        if re.compile("^[a-f0-9]{32}$").match(conversation_id) is None:
            raise ValueError(f"Incorrect conversation ID {conversation_id}")

    @staticmethod
    def construct_key(user_id: str, conversation_id: str) -> str:
        """Construct key to cache."""
        Cache._check_user_id(user_id)
        Cache._check_conversation_id(conversation_id)
        return f"{user_id}/{conversation_id}"

    @abstractmethod
    def get(self, user_id: str, conversation_id: str) -> Union[str, None]:
        """Abstract method to retrieve a value from the cache.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.

        Returns:
            The value associated with the key, or None if not found.
        """

    @abstractmethod
    def insert_or_append(self, user_id: str, conversation_id: str, value: str) -> None:
        """Abstract method to store a value in the cache.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            value: The value to be stored in the cache.
        """
