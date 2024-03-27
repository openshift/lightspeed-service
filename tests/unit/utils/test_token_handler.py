"""Unit test for the token handler."""

from typing import Any
from unittest import TestCase, mock

from langchain_core.messages import AIMessage, HumanMessage

from ols.utils.token_handler import TokenHandler


class MockRetrievedNode:
    """Class to create mock data for retrieved node."""

    def __init__(self, node_detail: dict):
        """Initialize the class instance."""
        self._text = node_detail["text"]
        self._score = node_detail["score"]
        self._metadata = node_detail["metadata"]

    def get_text(self) -> str:
        """Mock get_text."""
        return self._text

    @property
    def score(self) -> float:
        """Mock score."""
        return self._score

    @property
    def metadata(self) -> dict[str, Any]:
        """Mock metadata."""
        return self._metadata


class TestTokenHandler(TestCase):
    """Test cases for TokenHandler."""

    def setUp(self):
        """Set up mock data."""
        node_data = [
            {
                "text": "a text text text text",
                "score": 0.6,
                "metadata": {"docs_url": "data/doc1.pdf"},
            },
            {
                "text": "b text text text text",
                "score": 0.55,
                "metadata": {"docs_url": "data/doc2.pdf"},
            },
            {
                "text": "c text text text text",
                "score": 0.55,
                "metadata": {"docs_url": "data/doc3.pdf"},
            },
            {
                "text": "d text text text text",
                "score": 0.4,
                "metadata": {"docs_url": "data/doc4.pdf"},
            },
        ]
        self._mock_retrieved_obj = [MockRetrievedNode(data) for data in node_data]
        self._token_handler_obj = TokenHandler()

    def test_token_handler(self):
        """Test token handler for context."""
        retrieved_nodes = self._mock_retrieved_obj[:3]
        context = self._token_handler_obj.truncate_rag_context(retrieved_nodes)

        assert len(context) == len(retrieved_nodes)
        for idx, data in enumerate(context):
            assert data["text"][0] == self._mock_retrieved_obj[idx].get_text()[0]
            assert (
                data["docs_url"] == self._mock_retrieved_obj[idx].metadata["docs_url"]
            )

    def test_token_handler_token_limit(self):
        """Test token handler when token limit is reached."""
        context = self._token_handler_obj.truncate_rag_context(
            self._mock_retrieved_obj, 7
        )

        assert len(context) == 2
        assert (
            context[1]["text"].split()
            == self._mock_retrieved_obj[1].get_text().split()[:2]
        )

    @mock.patch("ols.utils.token_handler.MINIMUM_CONTEXT_LIMIT", 3)
    def test_token_handler_token_minimum(self):
        """Test token handler when token count reached minimum threshold."""
        context = self._token_handler_obj.truncate_rag_context(
            self._mock_retrieved_obj, 7
        )

        assert len(context) == 1

    def test_token_handler_empty(self):
        """Test token handler when node is empty."""
        context = self._token_handler_obj.truncate_rag_context([], 5)

        assert len(context) == 0

    def test_message_length_string_content(self):
        """Test the message_length method when message content is a string."""
        message = HumanMessage(content="")
        context = self._token_handler_obj.message_to_tokens(message)
        assert context == []

        message = HumanMessage(content="message from human")
        context = self._token_handler_obj.message_to_tokens(message)
        assert len(context) == 3

        message = AIMessage(content="message from AI system")
        context = self._token_handler_obj.message_to_tokens(message)
        assert len(context) == 4

    def test_message_length_list_content(self):
        """Test the message_length method when message content is list of strings."""
        message = HumanMessage(content=[""])
        context = self._token_handler_obj.message_to_tokens(message)
        assert context == []

        message = HumanMessage(content=["message from", "human"])
        context = self._token_handler_obj.message_to_tokens(message)
        assert len(context) == 3

        message = AIMessage(content=["message from", "AI system"])
        context = self._token_handler_obj.message_to_tokens(message)
        assert len(context) == 4

    def test_limit_conversation_history_when_no_history_exists(self):
        """Check the behaviour of limiting conversation history if it does not exists."""
        history, truncated = TokenHandler.limit_conversation_history([], 1000)
        # history must be empty
        assert history == []
        assert not truncated

    def test_limit_short_conversation_history(self):
        """Check the behaviour of limiting short conversation history."""
        history = [
            HumanMessage(content="first message from human"),
            AIMessage(content="first answer from AI"),
            HumanMessage(content="second message from human"),
            AIMessage(content="second answer from AI"),
        ]
        truncated_history, truncated = TokenHandler.limit_conversation_history(
            history, 1000
        )
        # history must remain the same and truncate flag should be False
        assert truncated_history == history
        assert not truncated

    def test_limit_long_conversation_history(self):
        """Check the behaviour of limiting long conversation history."""
        history = [
            HumanMessage(content="first message from human"),
            AIMessage(content="first answer from AI"),
            HumanMessage(content="second message from human"),
            AIMessage(content="second answer from AI"),
            HumanMessage(content="third message from human"),
            AIMessage(content="third answer from AI"),
        ]

        # try to truncate to 16 tokens
        truncated_history, truncated = TokenHandler.limit_conversation_history(
            history, 16
        )
        # history should truncate to 4 newest messages only and flag should be True
        assert len(truncated_history) == 4
        assert truncated_history == history[2:]
        assert truncated

        # try to truncate to 8 tokens
        truncated_history, truncated = TokenHandler.limit_conversation_history(
            history, 8
        )
        # history should truncate to 2 messages only and flag should be True
        assert len(truncated_history) == 2
        assert truncated_history == history[4:]
        assert truncated

        # try to truncate to 4 tokens - this means just one message
        truncated_history, truncated = TokenHandler.limit_conversation_history(
            history, 4
        )
        # history should truncate to one message only and flag should be True
        assert len(truncated_history) == 1
        assert truncated_history == history[5:]
        assert truncated

        # try to truncate to zero tokens
        truncated_history, truncated = TokenHandler.limit_conversation_history(
            history, 0
        )
        # history should truncate to empty list and flag should be True
        assert truncated_history == []
        assert truncated

        # try to truncate to one token, but the 1st message is already longer than 1 token
        truncated_history, truncated = TokenHandler.limit_conversation_history(
            history, 1
        )
        # history should truncate to empty list and flag should be True
        assert truncated_history == []
        assert truncated
