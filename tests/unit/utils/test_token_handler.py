"""Unit test for the token handler."""

from typing import Any
from unittest import TestCase, mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ols.app.models.config import ModelConfig
from ols.utils.token_handler import PromptTooLongError, TokenHandler


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
                "score": 0.4,
                "metadata": {"docs_url": "data/doc1.pdf", "title": "Doc1"},
            },
            {
                "text": "b text text text text",
                "score": 0.55,
                "metadata": {"docs_url": "data/doc2.pdf", "title": "Doc2"},
            },
            {
                "text": "c text text text text",
                "score": 0.55,
                "metadata": {"docs_url": "data/doc3.pdf", "title": "Doc3"},
            },
            {
                "text": "d text text text text",
                "score": 0.6,
                "metadata": {"docs_url": "data/doc4.pdf", "title": "Doc4"},
            },
        ]
        self._mock_retrieved_obj = [MockRetrievedNode(data) for data in node_data]
        self._token_handler_obj = TokenHandler()

    def test_available_tokens_for_empty_prompt(self):
        """Test the get_available_tokens method for default model config."""
        # use default model config
        model_config = ModelConfig(name="test")

        prompt = ""

        available_tokens = self._token_handler_obj.get_available_tokens(
            prompt, model_config
        )
        assert (
            available_tokens
            == model_config.context_window_size - model_config.response_token_limit
        )

    def test_available_tokens_for_regular_prompt(self):
        """Test the get_available_tokens method for default model config."""
        # use default model config
        model_config = ModelConfig(name="test")

        prompt = "What is Kubernetes?"
        prompt_length = len(self._token_handler_obj.text_to_tokens(prompt))

        available_tokens = self._token_handler_obj.get_available_tokens(
            prompt, model_config
        )
        expected_value = (
            model_config.context_window_size
            - model_config.response_token_limit
            - prompt_length
        )
        assert available_tokens == expected_value

    def test_available_tokens_for_large_prompt(self):
        """Test the get_available_tokens method for default model config."""
        # use default model config
        model_config = ModelConfig(name="test")

        # this prompt will surely exceeds context window size
        prompt = "What is Kubernetes?" * 10000

        with pytest.raises(
            PromptTooLongError, match="Prompt length exceeds LLM context window limit"
        ):
            self._token_handler_obj.get_available_tokens(prompt, model_config)

    def test_available_tokens_specific_model_config(self):
        """Test the get_available_tokens method for specific model config."""
        # use specific model config
        model_config = ModelConfig(
            **{
                "name": "test_name",
                "url": "http://test_url.com",
                "context_window_size": 100,
                "response_token_limit": 50,
            },
        )

        prompt = "What is Kubernetes?"
        prompt_length = len(self._token_handler_obj.text_to_tokens(prompt))

        available_tokens = self._token_handler_obj.get_available_tokens(
            prompt, model_config
        )
        expected_value = (
            model_config.context_window_size
            - model_config.response_token_limit
            - prompt_length
        )
        assert available_tokens == expected_value

    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF_L2", 0.9)
    def test_token_handler(self):
        """Test token handler for context."""
        retrieved_nodes = self._mock_retrieved_obj[:3]
        context, available_tokens = self._token_handler_obj.truncate_rag_context(
            retrieved_nodes
        )

        assert len(context) == len(("text", "docs_url", "title"))
        assert len(context["text"]) == 3
        for idx in range(3):
            assert context["text"][idx] == self._mock_retrieved_obj[idx].get_text()
            assert (
                context["docs_url"][idx]
                == self._mock_retrieved_obj[idx].metadata["docs_url"]
            )
        assert available_tokens == 485

    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF_L2", 0.5)
    def test_token_handler_score(self):
        """Test token handler for context when score is higher than threshold."""
        retrieved_nodes = self._mock_retrieved_obj[:3]
        context, available_tokens = self._token_handler_obj.truncate_rag_context(
            retrieved_nodes
        )

        assert len(context) == len(("text", "docs_url", "title"))
        assert len(context["text"]) == 1
        assert context["text"][0] == self._mock_retrieved_obj[0].get_text()
        assert available_tokens == 495

    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF_L2", 0.9)
    def test_token_handler_token_limit(self):
        """Test token handler when token limit is reached."""
        context, available_tokens = self._token_handler_obj.truncate_rag_context(
            self._mock_retrieved_obj, 7
        )

        assert len(context) == 3
        assert len(context["text"]) == 2
        assert (
            context["text"][1].split()
            == self._mock_retrieved_obj[1].get_text().split()[:2]
        )
        assert available_tokens == 0

    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF_L2", 0.9)
    @mock.patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3)
    def test_token_handler_token_minimum(self):
        """Test token handler when token count reached minimum threshold."""
        context, available_tokens = self._token_handler_obj.truncate_rag_context(
            self._mock_retrieved_obj, 7
        )

        assert len(context["text"]) == 1
        assert available_tokens == 2

    def test_token_handler_empty(self):
        """Test token handler when node is empty."""
        context, available_tokens = self._token_handler_obj.truncate_rag_context([], 5)

        assert len(context) == 0
        assert isinstance(context, dict)
        assert available_tokens == 5

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
