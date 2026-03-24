"""Unit test for the token handler."""

from math import ceil
from unittest import TestCase, mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ols.constants import TOKEN_BUFFER_WEIGHT
from ols.utils.token_handler import PromptTooLongError, TokenBudget, TokenHandler
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


class TestMeasureMessage:
    """Test cases for TokenHandler.measure_message."""

    def setup_method(self):
        """Set up token handler instance."""
        self._handler = TokenHandler()

    def test_measure_human_message(self):
        """Test that HumanMessage is measured as type: content."""
        msg = HumanMessage("hello world")
        count = self._handler.measure_message(msg)
        expected = TokenHandler._get_token_count(
            self._handler.text_to_tokens("human: hello world")
        )
        assert count == expected

    def test_measure_ai_message_without_tool_calls(self):
        """Test that AIMessage without tool_calls is measured as type: content."""
        msg = AIMessage("some response")
        count = self._handler.measure_message(msg)
        expected = TokenHandler._get_token_count(
            self._handler.text_to_tokens("ai: some response")
        )
        assert count == expected

    def test_measure_ai_message_with_tool_calls(self):
        """Test that AIMessage with tool_calls includes serialized tool_calls."""
        tool_calls = [{"name": "get_pods", "args": {"ns": "default"}, "id": "c1"}]
        msg = AIMessage(content="", tool_calls=tool_calls)
        count = self._handler.measure_message(msg)
        count_without = self._handler.measure_message(AIMessage(content=""))
        assert count > count_without

    def test_measure_tool_message(self):
        """Test that ToolMessage includes tool_call_id in measurement."""
        msg = ToolMessage(content="result data", tool_call_id="call_123")
        count = self._handler.measure_message(msg)
        plain_count = TokenHandler._get_token_count(
            self._handler.text_to_tokens("tool: result data")
        )
        assert count > plain_count

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.05)
    def test_measure_message_matches_legacy_for_plain_messages(self):
        """Verify measure_message produces the same count as the legacy inline formula."""
        handler = TokenHandler()
        for msg in [HumanMessage("test"), AIMessage("test")]:
            legacy = TokenHandler._get_token_count(
                handler.text_to_tokens(f"{msg.type}: {msg.content}")
            )
            assert handler.measure_message(msg) == legacy


class TestTokenBudget:
    """Test cases for TokenBudget."""

    def test_initial_available_for_augmentation(self):
        """Test available_for_augmentation before prompt overhead is set."""
        budget = TokenBudget(TokenHandler(), 1000, 200, 100)
        assert budget.available_for_augmentation == 700

    def test_set_prompt_overhead_returns_available(self):
        """Test set_prompt_overhead returns available tokens for augmentation."""
        budget = TokenBudget(TokenHandler(), 1000, 200, 100)
        available = budget.set_prompt_overhead("")
        assert available == 700

    def test_set_prompt_overhead_reduces_available(self):
        """Test that a non-empty prompt reduces available tokens."""
        budget = TokenBudget(TokenHandler(), 1000, 200, 100)
        available = budget.set_prompt_overhead("This is a test prompt")
        assert available < 700
        assert available > 0

    def test_set_prompt_overhead_raises_on_overflow(self):
        """Test PromptTooLongError when prompt exceeds context window."""
        budget = TokenBudget(TokenHandler(), 100, 50, 30)
        with pytest.raises(PromptTooLongError):
            budget.set_prompt_overhead("word " * 100)

    def test_remaining_tool_budget_initial(self):
        """Test initial remaining tool budget equals tool reserve."""
        budget = TokenBudget(TokenHandler(), 1000, 200, 500)
        assert budget.remaining_tool_budget == 500

    def test_consume_tool_tokens_reduces_remaining(self):
        """Test that consuming tool tokens reduces remaining budget."""
        budget = TokenBudget(TokenHandler(), 1000, 200, 500)
        budget.consume_tool_tokens(100)
        assert budget.remaining_tool_budget == 400
        assert budget.tool_tokens_used == 100
        budget.consume_tool_tokens(50)
        assert budget.remaining_tool_budget == 350
        assert budget.tool_tokens_used == 150

    def test_zero_tool_reserve(self):
        """Test budget with no tool reserve (no MCP servers)."""
        budget = TokenBudget(TokenHandler(), 1000, 200)
        assert budget.remaining_tool_budget == 0
        assert budget.available_for_augmentation == 800

    def test_set_prompt_overhead_overwrites_previous(self):
        """Test that calling set_prompt_overhead twice replaces the recorded value."""
        budget = TokenBudget(TokenHandler(), 1000, 200)
        first = budget.set_prompt_overhead("short")
        second = budget.set_prompt_overhead("a much longer prompt with many words")
        assert second < first
