"""Produce chat history object using list of Conversations."""

from typing import Optional

from langchain.schema import ChatMessage as langchainCM
from llama_index.llms import ChatMessage, MessageRole

from ols.src.cache.conversation import Conversation


def get_llama_index_chat_history(
    conversations: Optional[list[Conversation]],
) -> list[ChatMessage]:
    """Produce list of ChatMessage object using list of Conversations for llama_index."""
    chat_history: list[Conversation] = []
    if conversations is None or len(conversations) < 1:
        return chat_history
    for conversation in conversations:
        user_chat_message = ChatMessage(
            role=MessageRole.USER, content=conversation.user_message
        )
        chat_history.append(user_chat_message)
        assistant_chat_message = ChatMessage(
            role=MessageRole.ASSISTANT, content=conversation.assistant_message
        )
        chat_history.append(assistant_chat_message)
    return chat_history


def get_langchain_chat_history(
    conversations: Optional[list[Conversation]],
) -> list[langchainCM]:
    """Produce list of ChatMessage object using list of Conversations for langchain."""
    chat_history: list[Conversation] = []
    if conversations is None or len(conversations) < 1:
        return chat_history
    for conversation in conversations:
        user_chat_message = langchainCM(
            role=MessageRole.USER, content=conversation.user_message
        )
        chat_history.append(user_chat_message)
        assistant_chat_message = langchainCM(
            role=MessageRole.ASSISTANT, content=conversation.assistant_message
        )
        chat_history.append(assistant_chat_message)
    return chat_history
