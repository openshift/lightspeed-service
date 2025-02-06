"""Unit test for the token handler."""

from math import ceil
from unittest import TestCase, mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ols.constants import TOKEN_BUFFER_WEIGHT, ModelFamily
from ols.src.prompts.prompt_generator import restructure_history
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
        """Test the method to calculate available tokens and check if there are any available tokens for default model config."""  # noqa: E501
        context_window_size = 500
        max_tokens_for_response = 20

        prompt = ""

        available_tokens = self._token_handler_obj.calculate_and_check_available_tokens(
            prompt, context_window_size, max_tokens_for_response
        )
        assert available_tokens == context_window_size - max_tokens_for_response

    def test_available_tokens_for_regular_prompt(self):
        """Test the method to calculate available tokens and check if there are any available tokens for default model config."""  # noqa: E501
        context_window_size = 500
        max_tokens_for_response = 20

        prompt = "What is Kubernetes?"
        prompt_length = len(self._token_handler_obj.text_to_tokens(prompt))

        available_tokens = self._token_handler_obj.calculate_and_check_available_tokens(
            prompt, context_window_size, max_tokens_for_response
        )
        expected_value = (
            context_window_size
            - max_tokens_for_response
            - ceil(prompt_length * TOKEN_BUFFER_WEIGHT)
        )
        assert available_tokens == expected_value

    def test_available_tokens_for_large_prompt(self):
        """Test the method to calculate available tokens and check if there are any available tokens for default model config."""  # noqa: E501
        context_window_size = 500
        max_tokens_for_response = 20
        context_limit = context_window_size - max_tokens_for_response

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
                prompt, context_window_size, max_tokens_for_response
            )

    def test_available_tokens_with_buffer_weight(self):
        """Test the method to calculate available tokens and check if there are any available tokens for specific model config."""  # noqa: E501
        context_window_size = 500
        max_tokens_for_response = 20

        prompt = "What is Kubernetes?"
        prompt_length = len(self._token_handler_obj.text_to_tokens(prompt))

        available_tokens = self._token_handler_obj.calculate_and_check_available_tokens(
            prompt, context_window_size, max_tokens_for_response
        )
        expected_value = (
            context_window_size
            - max_tokens_for_response
            - ceil(prompt_length * TOKEN_BUFFER_WEIGHT)
        )
        assert available_tokens == expected_value

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.05)
    @mock.patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 1)
    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4)
    def test_token_handler(self):
        """Test token handler for context."""
        retrieved_nodes = self._mock_retrieved_obj[:3]
        rag_chunks, available_tokens = self._token_handler_obj.truncate_rag_context(
            retrieved_nodes, ModelFamily.GPT
        )

        assert len(rag_chunks) == 3
        for i in range(3):
            # Additionally tag and new line are added to have proper token calculation.
            assert (
                rag_chunks[i].text
                == "\nDocument:\n" + self._mock_retrieved_obj[i].get_text() + "\n"
            )
            assert (
                rag_chunks[i].doc_url
                == self._mock_retrieved_obj[i].metadata["docs_url"]
            )
        assert available_tokens == 473

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.05)
    @mock.patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3)
    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.6)
    def test_token_handler_score(self):
        """Test token handler for context when score is higher than threshold."""
        retrieved_nodes = self._mock_retrieved_obj[:3]
        rag_chunks, available_tokens = self._token_handler_obj.truncate_rag_context(
            retrieved_nodes, ModelFamily.GPT
        )

        assert len(rag_chunks) == 1
        assert (
            rag_chunks[0].text
            == "\nDocument:\n" + self._mock_retrieved_obj[0].get_text() + "\n"
        )
        assert available_tokens == 491

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.05)
    @mock.patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 4)
    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4)
    def test_token_handler_token_limit(self):
        """Test token handler when token limit is reached."""
        rag_chunks, available_tokens = self._token_handler_obj.truncate_rag_context(
            self._mock_retrieved_obj, ModelFamily.GPT, 13
        )

        assert len(rag_chunks) == 2
        assert (
            rag_chunks[1].text
            == "\nDocument:\n" + self._mock_retrieved_obj[1].get_text()[:1] + "\n"
        )
        assert available_tokens == 0

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.05)
    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4)
    @mock.patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3)
    def test_token_handler_token_minimum(self):
        """Test token handler when token count reached minimum threshold."""
        rag_chunks, available_tokens = self._token_handler_obj.truncate_rag_context(
            self._mock_retrieved_obj, ModelFamily.GPT, 10
        )

        assert len(rag_chunks) == 1
        assert available_tokens == 1

    def test_token_handler_empty(self):
        """Test token handler when node is empty."""
        rag_chunks, available_tokens = self._token_handler_obj.truncate_rag_context(
            [], ModelFamily.GRANITE, 5
        )

        assert rag_chunks == []
        assert available_tokens == 5

    def test_limit_conversation_history_when_no_history_exists(self):
        """Check the behaviour of limiting conversation history if it does not exists."""
        history, truncated = self._token_handler_obj.limit_conversation_history(
            [], ModelFamily.GPT, 1000
        )
        # history must be empty
        assert history == []
        assert not truncated

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.05)
    def test_limit_conversation_history(self):
        """Check the behaviour of limiting long conversation history."""
        history = [
            HumanMessage("first message from human"),
            AIMessage("first answer from AI"),
            HumanMessage("second message from human"),
            AIMessage("second answer from AI"),
            HumanMessage("third message from human"),
            AIMessage("third answer from AI"),
        ]
        # for each of the above actual messages the tokens count is 4.
        # then 2 tokens for the tags. Total tokens are 6.
        # As tokens are increased by 5% (ceil), final count becomes 7 per message.

        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(
                history, ModelFamily.GPT, 1000
            )
        )
        # history must remain the same and truncate flag should be False
        assert truncated_history == history
        assert not truncated

        # try to truncate to 28 tokens
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(
                history, ModelFamily.GPT, 28
            )
        )
        # history should truncate to 4 newest messages only and flag should be True
        assert len(truncated_history) == 4
        assert truncated_history == history[2:]
        assert truncated

        # try to truncate to 14 tokens
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(
                history, ModelFamily.GPT, 14
            )
        )
        # history should truncate to 2 messages only and flag should be True
        assert len(truncated_history) == 2
        assert truncated_history == history[4:]
        assert truncated

        # try to truncate to 13 tokens
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(
                history, ModelFamily.GPT, 13
            )
        )
        # history should truncate to 1 message
        assert len(truncated_history) == 1
        assert truncated_history == history[5:]
        assert truncated

        # try to truncate to 7 tokens - this means just one message
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(
                history, ModelFamily.GPT, 7
            )
        )
        # history should truncate to one message only and flag should be True
        assert len(truncated_history) == 1
        assert truncated_history == history[5:]
        assert truncated

        # try to truncate to zero tokens
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(
                history, ModelFamily.GPT, 0
            )
        )
        # history should truncate to empty list and flag should be True
        assert truncated_history == []
        assert truncated

        # try to truncate to one token, but the 1st message is already longer than 1 token
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(
                history, ModelFamily.GPT, 1
            )
        )
        # history should truncate to empty list and flag should be True
        assert truncated_history == []
        assert truncated

        # Test formatted history for granite
        model = ModelFamily.GRANITE
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(history, model, 22)
        )
        assert len(truncated_history) == 1
        assert truncated_history[-1] == restructure_history(history[-1], model)
        assert truncated
