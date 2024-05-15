"""Unit test for the token handler."""

from math import ceil
from unittest import TestCase, mock

import pytest

from ols.app.models.config import ModelConfig
from ols.constants import TOKEN_BUFFER_WEIGHT
from ols.utils.token_handler import PromptTooLongError, TokenHandler
from tests.mock_classes.mock_retrieved_node import MockRetrievedNode


class TestTokenHandler(TestCase):
    """Test cases for TokenHandler."""

    def setUp(self):
        """Set up mock data."""
        node_data = [
            {
                "text": "a text text text text",
                "score": 0.6,
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
                "score": 0.4,
                "metadata": {"docs_url": "data/doc4.pdf", "title": "Doc4"},
            },
        ]
        self._mock_retrieved_obj = [MockRetrievedNode(data) for data in node_data]
        self._token_handler_obj = TokenHandler()

    def test_available_tokens_for_empty_prompt(self):
        """Test the method to calculate available tokens and check if there are any available tokens for default model config."""  # noqa E501
        # use default model config
        model_config = ModelConfig({})

        prompt = ""

        available_tokens = self._token_handler_obj.calculate_and_check_available_tokens(
            prompt, model_config
        )
        assert (
            available_tokens
            == model_config.context_window_size - model_config.response_token_limit
        )

    def test_available_tokens_for_regular_prompt(self):
        """Test the method to calculate available tokens and check if there are any available tokens for default model config."""  # noqa E501
        # use default model config
        model_config = ModelConfig({})

        prompt = "What is Kubernetes?"
        prompt_length = len(self._token_handler_obj.text_to_tokens(prompt))

        available_tokens = self._token_handler_obj.calculate_and_check_available_tokens(
            prompt, model_config
        )
        expected_value = (
            model_config.context_window_size
            - model_config.response_token_limit
            - ceil(prompt_length * TOKEN_BUFFER_WEIGHT)
        )
        assert available_tokens == expected_value

    def test_available_tokens_for_large_prompt(self):
        """Test the method to calculate available tokens and check if there are any available tokens for default model config."""  # noqa E501
        # use default model config
        model_config = ModelConfig({})
        context_limit = (
            model_config.context_window_size - model_config.response_token_limit
        )

        # this prompt will surely exceeds context window size
        prompt = "What is Kubernetes?" * 10000
        prompt_length = len(self._token_handler_obj.text_to_tokens(prompt))
        prompt_length = ceil(prompt_length * TOKEN_BUFFER_WEIGHT)

        expected_error_messge = (
            f"Prompt length {prompt_length} exceeds "
            f"LLM available context window limit {context_limit} tokens"
        )
        with pytest.raises(PromptTooLongError, match=expected_error_messge):
            self._token_handler_obj.calculate_and_check_available_tokens(
                prompt, model_config
            )

    def test_available_tokens_specific_model_config(self):
        """Test the method to calculate available tokens and check if there are any available tokens for specific model config."""  # noqa E501
        # use specific model config
        model_config = ModelConfig(
            {
                "name": "test_name",
                "url": "test_url",
                "context_window_size": 100,
                "response_token_limit": 50,
            },
        )

        prompt = "What is Kubernetes?"
        prompt_length = len(self._token_handler_obj.text_to_tokens(prompt))

        available_tokens = self._token_handler_obj.calculate_and_check_available_tokens(
            prompt, model_config
        )
        expected_value = (
            model_config.context_window_size
            - model_config.response_token_limit
            - ceil(prompt_length * TOKEN_BUFFER_WEIGHT)
        )
        assert available_tokens == expected_value

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.05)
    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4)
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
        assert available_tokens == 482

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.05)
    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.6)
    def test_token_handler_score(self):
        """Test token handler for context when score is higher than threshold."""
        retrieved_nodes = self._mock_retrieved_obj[:3]
        context, available_tokens = self._token_handler_obj.truncate_rag_context(
            retrieved_nodes
        )

        assert len(context) == len(("text", "docs_url", "title"))
        assert len(context["text"]) == 1
        assert context["text"][0] == self._mock_retrieved_obj[0].get_text()
        assert available_tokens == 494

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.05)
    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4)
    def test_token_handler_token_limit(self):
        """Test token handler when token limit is reached."""
        context, available_tokens = self._token_handler_obj.truncate_rag_context(
            self._mock_retrieved_obj, 7
        )

        assert len(context) == 3
        assert len(context["text"]) == 2
        assert (
            context["text"][1].split()
            == self._mock_retrieved_obj[1].get_text().split()[:1]
        )
        assert available_tokens == 0

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.05)
    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4)
    @mock.patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3)
    def test_token_handler_token_minimum(self):
        """Test token handler when token count reached minimum threshold."""
        context, available_tokens = self._token_handler_obj.truncate_rag_context(
            self._mock_retrieved_obj, 7
        )

        assert len(context["text"]) == 1
        assert available_tokens == 1

    def test_token_handler_empty(self):
        """Test token handler when node is empty."""
        context, available_tokens = self._token_handler_obj.truncate_rag_context([], 5)

        assert len(context) == 0
        assert isinstance(context, dict)
        assert available_tokens == 5

    def test_limit_conversation_history_when_no_history_exists(self):
        """Check the behaviour of limiting conversation history if it does not exists."""
        history, truncated = self._token_handler_obj.limit_conversation_history(
            [], 1000
        )
        # history must be empty
        assert history == []
        assert not truncated

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.05)
    def test_limit_conversation_history(self):
        """Check the behaviour of limiting long conversation history."""
        history = [
            "first message from human",
            "first answer from AI",
            "second message from human",
            "second answer from AI",
            "third message from human",
            "third answer from AI",
        ]
        # As tokens are increased by 5% (ceil),
        # for each of the above messages the tokens count is 5, instead of 4.

        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(history, 1000)
        )
        # history must remain the same and truncate flag should be False
        assert truncated_history == history
        assert not truncated

        # try to truncate to 23 tokens; 20 for 4 messages & 3 for 3 new-lines.
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(history, 23)
        )
        # history should truncate to 4 newest messages only and flag should be True
        assert len(truncated_history) == 4
        assert truncated_history == history[2:]
        assert truncated

        # try to truncate to 11 tokens
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(history, 11)
        )
        # history should truncate to 2 messages only and flag should be True
        assert len(truncated_history) == 2
        assert truncated_history == history[4:]
        assert truncated

        # try to truncate to 10 tokens; without new line token
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(history, 10)
        )
        # history should truncate to 1 message
        assert len(truncated_history) == 1
        assert truncated_history == history[5:]
        assert truncated

        # try to truncate to 5 tokens - this means just one message
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(history, 5)
        )
        # history should truncate to one message only and flag should be True
        assert len(truncated_history) == 1
        assert truncated_history == history[5:]
        assert truncated

        # try to truncate to zero tokens
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(history, 0)
        )
        # history should truncate to empty list and flag should be True
        assert truncated_history == []
        assert truncated

        # try to truncate to one token, but the 1st message is already longer than 1 token
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(history, 1)
        )
        # history should truncate to empty list and flag should be True
        assert truncated_history == []
        assert truncated
