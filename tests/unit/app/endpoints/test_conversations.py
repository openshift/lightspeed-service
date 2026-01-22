"""Unit tests for conversations endpoint handlers."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from ols import config
from ols.app.models.models import (
    CacheEntry,
    ConversationData,
    ConversationUpdateRequest,
)

# needs to be setup before conversations endpoint is imported
config.ols_config.authentication_config.module = "k8s"

from ols.app.endpoints import conversations  # noqa: E402


@pytest.fixture
def mock_auth():
    """Create a mock auth tuple."""
    return ("test-user-id", "test-username", False, "test-token")


@pytest.fixture
def mock_auth_skip_check():
    """Create a mock auth tuple with skip_user_id_check=True."""
    return ("test-user-id", "test-username", True, "test-token")


@pytest.fixture
def mock_cache():
    """Create a mock cache."""
    return MagicMock()


@pytest.fixture
def sample_conversation_data():
    """Create sample conversation data."""
    return [
        ConversationData(
            conversation_id="123e4567-e89b-12d3-a456-426614174000",
            topic_summary="Test topic",
            last_message_timestamp=1737370502.0,
            message_count=2,
        ),
        ConversationData(
            conversation_id="223e4567-e89b-12d3-a456-426614174001",
            topic_summary="Another topic",
            last_message_timestamp=1737370600.0,
            message_count=5,
        ),
    ]


class TestListConversations:
    """Tests for list_conversations endpoint."""

    def test_list_conversations_success(
        self, mock_auth, mock_cache, sample_conversation_data
    ):
        """Test successful listing of conversations."""
        mock_cache.list.return_value = sample_conversation_data

        with patch("ols.config._conversation_cache", mock_cache):
            response = conversations.list_conversations(mock_auth)

        assert len(response.conversations) == 2
        assert (
            response.conversations[0].conversation_id
            == sample_conversation_data[0].conversation_id
        )
        mock_cache.list.assert_called_once_with("test-user-id", False)

    def test_list_conversations_empty(self, mock_auth, mock_cache):
        """Test listing conversations when none exist."""
        mock_cache.list.return_value = []

        with patch("ols.config._conversation_cache", mock_cache):
            response = conversations.list_conversations(mock_auth)

        assert len(response.conversations) == 0

    def test_list_conversations_error(self, mock_auth, mock_cache):
        """Test error handling when listing conversations fails."""
        mock_cache.list.side_effect = Exception("Database error")

        with patch("ols.config._conversation_cache", mock_cache):
            with pytest.raises(HTTPException) as exc_info:
                conversations.list_conversations(mock_auth)

        assert exc_info.value.status_code == 500
        assert "Error listing conversations" in exc_info.value.detail["response"]


class TestGetConversation:
    """Tests for get_conversation endpoint."""

    def test_get_conversation_success(self, mock_auth, mock_cache):
        """Test successful retrieval of a conversation."""
        mock_cache_entries = [
            CacheEntry(
                query=HumanMessage(content="Hello"),
                response=AIMessage(content="Hi there!"),
            ),
            CacheEntry(
                query=HumanMessage(content="How are you?"),
                response=AIMessage(content="I'm doing well!"),
            ),
        ]
        mock_cache.get.return_value = mock_cache_entries
        conversation_id = "123e4567-e89b-12d3-a456-426614174000"

        with patch("ols.config._conversation_cache", mock_cache):
            response = conversations.get_conversation(conversation_id, mock_auth)

        assert response.conversation_id == conversation_id
        assert len(response.chat_history) == 2
        assert response.chat_history[0]["messages"][0]["type"] == "user"
        assert response.chat_history[0]["messages"][0]["content"] == "Hello"
        assert response.chat_history[0]["messages"][1]["type"] == "assistant"
        assert response.chat_history[0]["messages"][1]["content"] == "Hi there!"

    def test_get_conversation_not_found(self, mock_auth, mock_cache):
        """Test retrieval of non-existent conversation."""
        mock_cache.get.return_value = []
        conversation_id = "123e4567-e89b-12d3-a456-426614174000"

        with patch("ols.config._conversation_cache", mock_cache):
            with pytest.raises(HTTPException) as exc_info:
                conversations.get_conversation(conversation_id, mock_auth)

        assert exc_info.value.status_code == 404
        assert "Conversation not found" in exc_info.value.detail["response"]

    def test_get_conversation_invalid_id(self, mock_auth, mock_cache):
        """Test retrieval with invalid conversation ID."""
        with pytest.raises(HTTPException) as exc_info:
            conversations.get_conversation("invalid-id", mock_auth)

        assert exc_info.value.status_code == 400
        assert "Invalid conversation ID format" in exc_info.value.detail["response"]


class TestDeleteConversation:
    """Tests for delete_conversation endpoint."""

    def test_delete_conversation_success(self, mock_auth, mock_cache):
        """Test successful deletion of a conversation."""
        mock_cache.delete.return_value = True
        conversation_id = "123e4567-e89b-12d3-a456-426614174000"

        with patch("ols.config._conversation_cache", mock_cache):
            response = conversations.delete_conversation(conversation_id, mock_auth)

        assert response.conversation_id == conversation_id
        assert response.success is True
        assert response.response == "Conversation deleted successfully"
        mock_cache.delete.assert_called_once_with(
            "test-user-id", conversation_id, False
        )

    def test_delete_conversation_not_found(self, mock_auth, mock_cache):
        """Test deletion of non-existent conversation."""
        mock_cache.delete.return_value = False
        conversation_id = "123e4567-e89b-12d3-a456-426614174000"

        with patch("ols.config._conversation_cache", mock_cache):
            response = conversations.delete_conversation(conversation_id, mock_auth)

        assert response.conversation_id == conversation_id
        assert response.success is False
        assert response.response == "Conversation not found"

    def test_delete_conversation_invalid_id(self, mock_auth, mock_cache):
        """Test deletion with invalid conversation ID."""
        with pytest.raises(HTTPException) as exc_info:
            conversations.delete_conversation("invalid-id", mock_auth)

        assert exc_info.value.status_code == 400
        assert "Invalid conversation ID format" in exc_info.value.detail["response"]


class TestUpdateConversation:
    """Tests for update_conversation endpoint."""

    def test_update_conversation_success(self, mock_auth, mock_cache):
        """Test successful update of a conversation."""
        conversation_id = "123e4567-e89b-12d3-a456-426614174000"
        update_request = ConversationUpdateRequest(topic_summary="Updated Topic")

        with patch("ols.config._conversation_cache", mock_cache):
            response = conversations.update_conversation(
                conversation_id, update_request, mock_auth
            )

        assert response.conversation_id == conversation_id
        assert response.success is True
        assert response.message == "Topic summary updated successfully"
        mock_cache.set_topic_summary.assert_called_once_with(
            "test-user-id", conversation_id, "Updated Topic", False
        )

    def test_update_conversation_invalid_id(self, mock_auth, mock_cache):
        """Test update with invalid conversation ID."""
        update_request = ConversationUpdateRequest(topic_summary="Updated Topic")

        with pytest.raises(HTTPException) as exc_info:
            conversations.update_conversation("invalid-id", update_request, mock_auth)

        assert exc_info.value.status_code == 400
        assert "Invalid conversation ID format" in exc_info.value.detail["response"]

    def test_update_conversation_error(self, mock_auth, mock_cache):
        """Test error handling when update fails."""
        mock_cache.set_topic_summary.side_effect = Exception("Database error")
        conversation_id = "123e4567-e89b-12d3-a456-426614174000"
        update_request = ConversationUpdateRequest(topic_summary="Updated Topic")

        with patch("ols.config._conversation_cache", mock_cache):
            with pytest.raises(HTTPException) as exc_info:
                conversations.update_conversation(
                    conversation_id, update_request, mock_auth
                )

        assert exc_info.value.status_code == 500
        assert "Error updating conversation" in exc_info.value.detail["response"]


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_retrieve_user_id(self, mock_auth):
        """Test retrieving user ID from auth tuple."""
        user_id = conversations.retrieve_user_id(mock_auth)
        assert user_id == "test-user-id"

    def test_retrieve_skip_user_id_check(self, mock_auth, mock_auth_skip_check):
        """Test retrieving skip_user_id_check from auth tuple."""
        assert conversations.retrieve_skip_user_id_check(mock_auth) is False
        assert conversations.retrieve_skip_user_id_check(mock_auth_skip_check) is True
