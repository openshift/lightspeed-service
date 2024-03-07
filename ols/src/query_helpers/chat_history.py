"""Helper classes to get chat history objects for conversation."""

import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.base import BaseMessage

logger = logging.getLogger(__name__)


class ChatHistory:
    """Helper class to get chat history objects for conversation."""

    @staticmethod
    def get_chat_message_history(
        user_message: str, ai_response: str
    ) -> list[BaseMessage]:
        """Get chat message history object.

        Args:
            user_message: The user's message.
            ai_response: The AI's response.

        Returns:
            [ChatMessage]: List of ChatMessage objects with HumanMessage and AIMessage.
        """
        chat_history: list[BaseMessage] = []
        chat_history.append(HumanMessage(content=user_message))
        chat_history.append(AIMessage(content=ai_response))
        return chat_history
