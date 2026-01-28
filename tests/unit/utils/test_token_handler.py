"""Unit test for the token handler."""

from math import ceil
from unittest import TestCase, mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

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
            retrieved_nodes
        )

        assert len(rag_chunks) == 3
        for i in range(3):
            # New-line character is considered during calculation.
            assert (
                rag_chunks[i].text
                == "Document:\n" + self._mock_retrieved_obj[i].get_text()
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
            retrieved_nodes
        )
        assert len(rag_chunks) == 1
        assert (
            rag_chunks[0].text == "Document:\n" + self._mock_retrieved_obj[0].get_text()
        )
        assert available_tokens == 491

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.05)
    @mock.patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 4)
    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4)
    def test_token_handler_token_limit(self):
        """Test token handler when token limit is reached."""
        # Calculation for each chunk:
        # `Document:\n` -> 3 tokens for format, Actual text -> 5, new-line -> 1, total -> 9

        rag_chunks, available_tokens = self._token_handler_obj.truncate_rag_context(
            self._mock_retrieved_obj, 13
        )
        assert len(rag_chunks) == 2
        assert (
            rag_chunks[1].text
            == "Document:\n" + self._mock_retrieved_obj[1].get_text()[:6]
        )
        assert available_tokens == 0

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.05)
    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4)
    @mock.patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3)
    def test_token_handler_token_minimum(self):
        """Test token handler when token count reached minimum threshold."""
        rag_chunks, available_tokens = self._token_handler_obj.truncate_rag_context(
            self._mock_retrieved_obj, 10
        )
        assert len(rag_chunks) == 1
        assert available_tokens == 1

    def test_token_handler_empty(self):
        """Test token handler when node is empty."""
        rag_chunks, available_tokens = self._token_handler_obj.truncate_rag_context(
            [], 5
        )
        assert rag_chunks == []
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
            HumanMessage("first message from human"),
            AIMessage("first answer from AI"),
            HumanMessage("second message from human"),
            AIMessage("second answer from AI"),
            HumanMessage("third message from human"),
            AIMessage("third answer from AI"),
        ]
        # for each of the above actual messages the tokens counts as below
        # `role:` -> 2, Actual content -> 4, new-line -> 1, total -> 7
        # As tokens are increased by 5% (ceil), final count becomes 8 per message.

        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(history, 1000)
        )
        # history must remain the same and truncate flag should be False
        assert truncated_history == history
        assert not truncated

        # try to truncate to 28 tokens
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(history, 32)
        )
        # history should truncate to 4 newest messages only and flag should be True
        assert len(truncated_history) == 4
        assert truncated_history == history[2:]
        assert truncated

        # try to truncate to 14 tokens
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(history, 16)
        )
        # history should truncate to 2 messages only and flag should be True
        assert len(truncated_history) == 2
        assert truncated_history == history[4:]
        assert truncated

        # try to truncate to 13 tokens
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(history, 13)
        )
        # history should truncate to 1 message
        assert len(truncated_history) == 1
        assert truncated_history == history[5:]
        assert truncated

        # try to truncate to 7 tokens - this means just one message
        truncated_history, truncated = (
            self._token_handler_obj.limit_conversation_history(history, 8)
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

    def test_available_tokens_with_tool_reservation(self):
        """Test token calculation with reserved tokens for tools."""
        context_window_size = 500
        max_tokens_for_response = 20
        max_tokens_for_tools = 100

        prompt = ""

        # Without tool reservation
        available_without_tools = (
            self._token_handler_obj.calculate_and_check_available_tokens(
                prompt, context_window_size, max_tokens_for_response
            )
        )
        assert available_without_tools == context_window_size - max_tokens_for_response

        # With tool reservation
        available_with_tools = (
            self._token_handler_obj.calculate_and_check_available_tokens(
                prompt,
                context_window_size,
                max_tokens_for_response,
                max_tokens_for_tools,
            )
        )
        assert (
            available_with_tools
            == context_window_size - max_tokens_for_response - max_tokens_for_tools
        )

        # Verify the difference equals the tool reservation
        assert available_without_tools - available_with_tools == max_tokens_for_tools

    def test_available_tokens_tool_reservation_causes_overflow(self):
        """Test that tool reservation can cause prompt overflow error."""
        context_window_size = 100
        max_tokens_for_response = 20
        max_tokens_for_tools = 50

        # A prompt that fits without tool reservation but not with it
        # Need a prompt of ~35-40 tokens to fit in 80 (100-20) but not in 30 (100-20-50)
        prompt = "word " * 30  # roughly 30+ tokens with buffer

        # Should work without tool reservation (80 tokens available)
        available = self._token_handler_obj.calculate_and_check_available_tokens(
            prompt, context_window_size, max_tokens_for_response
        )
        assert available > 0

        # Should raise error with tool reservation (only 30 tokens available)
        with pytest.raises(PromptTooLongError):
            self._token_handler_obj.calculate_and_check_available_tokens(
                prompt,
                context_window_size,
                max_tokens_for_response,
                max_tokens_for_tools,
            )

    def test_truncate_tool_output_no_truncation_needed(self):
        """Test truncate_tool_output when output is within limit."""
        short_output = "This is a short tool output."

        result, was_truncated = self._token_handler_obj.truncate_tool_output(
            short_output, max_tokens=1000
        )

        assert result == short_output
        assert was_truncated is False

    def test_truncate_tool_output_truncation_needed(self):
        """Test truncate_tool_output when output exceeds limit."""
        # Create a long output that will exceed the limit
        long_output = "word " * 500  # roughly 500 tokens

        result, was_truncated = self._token_handler_obj.truncate_tool_output(
            long_output, max_tokens=100
        )

        assert was_truncated is True
        # Check that warning message is appended
        assert "[OUTPUT TRUNCATED" in result
        assert "Please ask a more specific question" in result
        # Result should be shorter than original
        assert len(result) < len(long_output)

    def test_truncate_tool_output_preserves_beginning(self):
        """Test that truncation keeps the beginning of the output."""
        # Create output with recognizable start
        long_output = "START_MARKER " + ("filler " * 500) + " END_MARKER"

        result, was_truncated = self._token_handler_obj.truncate_tool_output(
            long_output, max_tokens=100
        )

        assert was_truncated is True
        # Beginning should be preserved
        assert result.startswith("START_MARKER")
        # End should be truncated (replaced with warning)
        assert "END_MARKER" not in result

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.1)
    def test_truncate_tool_output_exact_limit(self):
        """Test truncate_tool_output when output is exactly at limit."""
        # Create output that's exactly at the weighted limit
        output = "test"
        tokens = self._token_handler_obj.text_to_tokens(output)
        # Account for buffer weight (ceil(len * 1.1))
        max_tokens = ceil(len(tokens) * 1.1)

        result, was_truncated = self._token_handler_obj.truncate_tool_output(
            output, max_tokens=max_tokens
        )

        assert result == output
        assert was_truncated is False
