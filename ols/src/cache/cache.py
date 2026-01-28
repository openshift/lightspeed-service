"""Abstract class that is parent for all cache implementations."""

from abc import ABC, abstractmethod

from ols.app.models.models import CacheEntry, ConversationData
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
    def _check_user_id(user_id: str, skip_user_id_check: bool) -> None:
        """Check if given user ID is valid."""
        if skip_user_id_check:
            return
        if not check_suid(user_id):
            raise ValueError(f"Invalid user ID {user_id}")

    @staticmethod
    def _check_conversation_id(conversation_id: str) -> None:
        """Check if given conversation ID is a valid UUID (including optional dashes)."""
        if not check_suid(conversation_id):
            raise ValueError(f"Invalid conversation ID {conversation_id}")

    @staticmethod
    def construct_key(
        user_id: str, conversation_id: str, skip_user_id_check: bool
    ) -> str:
        """Construct key to cache."""
        Cache._check_user_id(user_id, skip_user_id_check)
        Cache._check_conversation_id(conversation_id)
        return f"{user_id}{Cache.COMPOUND_KEY_SEPARATOR}{conversation_id}"

    @abstractmethod
    def get(
        self, user_id: str, conversation_id: str, skip_user_id_check: bool
    ) -> list[CacheEntry]:
        """Abstract method to retrieve a value from the cache.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            skip_user_id_check: Skip user_id suid check.

        Returns:
            The value (CacheEntry(s)) associated with the key, or None if not found.
        """

    @abstractmethod
    def insert_or_append(
        self,
        user_id: str,
        conversation_id: str,
        cache_entry: CacheEntry,
        skip_user_id_check: bool,
    ) -> None:
        """Abstract method to store a value in the cache.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            cache_entry: The value to store.
            skip_user_id_check: Skip user_id suid check.
        """

    @abstractmethod
    def delete(
        self, user_id: str, conversation_id: str, skip_user_id_check: bool
    ) -> bool:
        """Delete all entries for a given conversation.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            skip_user_id_check: Skip user_id suid check.

        Returns:
            bool: True if entries were deleted, False if key wasn't found.
        """

    @abstractmethod
    def list(self, user_id: str, skip_user_id_check: bool) -> list[ConversationData]:
        """List all conversations for a given user_id.

        Args:
            user_id: User identification.
            skip_user_id_check: Skip user_id suid check.

        Returns:
            A list of ConversationData objects containing conversation_id,
            topic_summary, last_message_timestamp, and message_count.
        """

    @abstractmethod
    def set_topic_summary(
        self,
        user_id: str,
        conversation_id: str,
        topic_summary: str,
        skip_user_id_check: bool,
    ) -> None:
        """Set or update the topic summary for a conversation.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            topic_summary: The topic summary to store.
            skip_user_id_check: Skip user_id suid check.
        """

    @abstractmethod
    def ready(self) -> bool:
        """Check if the cache is ready.

        Returns:
            True if the cache is ready, False otherwise.
        """
