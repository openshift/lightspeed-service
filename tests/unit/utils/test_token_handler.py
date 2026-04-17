"""Unit test for the token handler."""

from math import ceil
from unittest import TestCase, mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ols.constants import DEFAULT_TOOL_ROUND_CAP_FRACTION, TOKEN_BUFFER_WEIGHT
from ols.utils.token_handler import (
    PromptTooLongError,
    TokenBudgetTracker,
    TokenCategory,
    TokenHandler,
)
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
        rag_chunks = self._token_handler_obj.truncate_rag_context(retrieved_nodes)

        assert len(rag_chunks) == 3
        for i in range(3):
            assert (
                rag_chunks[i].text
                == "Document:\n" + self._mock_retrieved_obj[i].get_text()
            )
            assert (
                rag_chunks[i].doc_url
                == self._mock_retrieved_obj[i].metadata["docs_url"]
            )

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.05)
    @mock.patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3)
    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.6)
    def test_token_handler_score(self):
        """Test token handler for context when score is higher than threshold."""
        retrieved_nodes = self._mock_retrieved_obj[:3]
        rag_chunks = self._token_handler_obj.truncate_rag_context(retrieved_nodes)
        assert len(rag_chunks) == 1
        assert (
            rag_chunks[0].text == "Document:\n" + self._mock_retrieved_obj[0].get_text()
        )

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.05)
    @mock.patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 4)
    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4)
    def test_token_handler_token_limit(self):
        """Test token handler when token limit is reached."""
        # Calculation for each chunk:
        # `Document:\n` -> 3 tokens for format, Actual text -> 5, new-line -> 1, total -> 9

        rag_chunks = self._token_handler_obj.truncate_rag_context(
            self._mock_retrieved_obj, 13
        )
        assert len(rag_chunks) == 2
        assert (
            rag_chunks[1].text
            == "Document:\n" + self._mock_retrieved_obj[1].get_text()[:6]
        )

    @mock.patch("ols.utils.token_handler.TOKEN_BUFFER_WEIGHT", 1.05)
    @mock.patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4)
    @mock.patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3)
    def test_token_handler_token_minimum(self):
        """Test token handler when token count reached minimum threshold."""
        rag_chunks = self._token_handler_obj.truncate_rag_context(
            self._mock_retrieved_obj, 10
        )
        assert len(rag_chunks) == 1

    def test_token_handler_empty(self):
        """Test token handler when node is empty."""
        rag_chunks = self._token_handler_obj.truncate_rag_context([], 5)
        assert rag_chunks == []

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


class TestTokenBudgetTracker(TestCase):
    """Tests for TokenBudgetTracker budget properties."""

    def test_prompt_budget_matches_window_minus_reservations(self):
        """prompt_budget is context minus response and tool reservations."""
        tracker = TokenBudgetTracker(
            token_handler=TokenHandler(),
            context_window_size=1000,
            max_response_tokens=100,
            max_tool_tokens=200,
            round_cap_fraction=DEFAULT_TOOL_ROUND_CAP_FRACTION,
        )
        assert tracker.prompt_budget == 700

    def test_prompt_budget_remaining_reflects_charges(self):
        """prompt_budget_remaining decreases when categories accrue usage."""
        tracker = TokenBudgetTracker(
            token_handler=TokenHandler(),
            context_window_size=1000,
            max_response_tokens=100,
            max_tool_tokens=200,
            round_cap_fraction=DEFAULT_TOOL_ROUND_CAP_FRACTION,
        )
        assert tracker.prompt_budget_remaining == 700
        tracker.charge(TokenCategory.PROMPT, 150)
        assert tracker.prompt_budget_remaining == 550
        tracker.charge(TokenCategory.HISTORY, 50)
        assert tracker.prompt_budget_remaining == 500

    def test_remaining_initial_and_relation_to_prompt_budget(self):
        """Remaining is window minus response reservation minus all category usage."""
        tracker = TokenBudgetTracker(
            token_handler=TokenHandler(),
            context_window_size=1000,
            max_response_tokens=100,
            max_tool_tokens=200,
            round_cap_fraction=DEFAULT_TOOL_ROUND_CAP_FRACTION,
        )
        assert tracker.remaining == 900
        assert tracker.prompt_budget_remaining == 700
        assert (
            tracker.remaining
            == tracker.prompt_budget_remaining + tracker.max_tool_tokens
        )

    def test_remaining_decreases_with_prompt_and_tool_charges(self):
        """Any category charge lowers remaining by the same amount."""
        tracker = TokenBudgetTracker(
            token_handler=TokenHandler(),
            context_window_size=1000,
            max_response_tokens=100,
            max_tool_tokens=200,
            round_cap_fraction=DEFAULT_TOOL_ROUND_CAP_FRACTION,
        )
        assert tracker.remaining == 900
        tracker.charge(TokenCategory.PROMPT, 120)
        assert tracker.remaining == 780
        assert tracker.total_used == 120
        tracker.charge(TokenCategory.TOOL_RESULT, 40)
        assert tracker.remaining == 740
        assert tracker.total_used == 160

    def test_tool_budget_used_sums_only_tool_pool_categories(self):
        """tool_budget_used includes AI round and tool results, not definitions."""
        tracker = TokenBudgetTracker(
            token_handler=TokenHandler(),
            context_window_size=1000,
            max_response_tokens=100,
            max_tool_tokens=200,
            round_cap_fraction=DEFAULT_TOOL_ROUND_CAP_FRACTION,
        )
        tracker.charge(TokenCategory.TOOL_DEFINITIONS, 11)
        tracker.charge(TokenCategory.AI_ROUND, 22)
        tracker.charge(TokenCategory.TOOL_RESULT, 33)
        assert tracker.tool_budget_used == 55
        tracker.charge(TokenCategory.PROMPT, 400)
        tracker.charge(TokenCategory.HISTORY, 50)
        assert tracker.tool_budget_used == 55

    def test_tool_budget_remaining_clamps_at_zero(self):
        """tool_budget_remaining does not go negative when over the tool cap."""
        tracker = TokenBudgetTracker(
            token_handler=TokenHandler(),
            context_window_size=1000,
            max_response_tokens=100,
            max_tool_tokens=80,
            round_cap_fraction=DEFAULT_TOOL_ROUND_CAP_FRACTION,
        )
        tracker.charge(TokenCategory.TOOL_RESULT, 150)
        assert tracker.tool_budget_used == 150
        assert tracker.tool_budget_remaining == 0

    def test_tools_round_budget_scales_with_remaining_and_fraction(self):
        """tools_round_budget is int(fraction * tool_budget_remaining)."""
        tracker = TokenBudgetTracker(
            token_handler=TokenHandler(),
            context_window_size=1000,
            max_response_tokens=100,
            max_tool_tokens=100,
            round_cap_fraction=0.5,
        )
        assert tracker.tool_budget_remaining == 100
        assert tracker.tools_round_budget == 50
        tracker.charge(TokenCategory.TOOL_RESULT, 60)
        assert tracker.tool_budget_remaining == 40
        assert tracker.tools_round_budget == 20

    def test_tools_round_budget_truncates_with_int(self):
        """Fractional product uses int truncation toward zero."""
        tracker = TokenBudgetTracker(
            token_handler=TokenHandler(),
            context_window_size=1000,
            max_response_tokens=100,
            max_tool_tokens=100,
            round_cap_fraction=0.6,
        )
        tracker.charge(TokenCategory.TOOL_RESULT, 33)
        assert tracker.tool_budget_remaining == 67
        assert tracker.tools_round_budget == int(67 * 0.6)

    def test_tools_round_execution_budget_zero_when_tool_pool_exhausted(self):
        """tools_round_execution_budget is 0 when nothing remains in the tool slice."""
        tracker = TokenBudgetTracker(
            token_handler=TokenHandler(),
            context_window_size=1000,
            max_response_tokens=100,
            max_tool_tokens=50,
            round_cap_fraction=0.6,
        )
        tracker.charge(TokenCategory.TOOL_RESULT, 50)
        assert tracker.tools_round_execution_budget(5) == 0

    def test_tools_round_execution_budget_equal_share_tightens_long_horizon(self):
        """Long horizon makes equal-share smaller than fraction cap."""
        tracker = TokenBudgetTracker(
            token_handler=TokenHandler(),
            context_window_size=10_000,
            max_response_tokens=100,
            max_tool_tokens=10_000,
            round_cap_fraction=0.6,
        )
        assert tracker.tool_budget_remaining == 10_000
        assert tracker.tools_round_execution_budget(19) == min(
            int(10_000 * 0.6),
            10_000 // 19,
        )

    def test_tools_round_execution_budget_fraction_cap_when_horizon_is_one(self):
        """Single-round horizon leaves fraction cap as the tighter bound."""
        tracker = TokenBudgetTracker(
            token_handler=TokenHandler(),
            context_window_size=1000,
            max_response_tokens=100,
            max_tool_tokens=500,
            round_cap_fraction=0.6,
        )
        assert tracker.tools_round_execution_budget(1) == int(500 * 0.6)

    def test_tools_round_execution_budget_non_positive_horizon_treated_as_one(self):
        """tool_rounds_left below 1 uses horizon 1 (same as fraction-only for full R)."""
        tracker = TokenBudgetTracker(
            token_handler=TokenHandler(),
            context_window_size=1000,
            max_response_tokens=100,
            max_tool_tokens=200,
            round_cap_fraction=0.5,
        )
        assert tracker.tools_round_execution_budget(
            0
        ) == tracker.tools_round_execution_budget(1)
        assert tracker.tools_round_execution_budget(-3) == 100

    def test_summary_includes_tools_exec_budget_after_set_tool_loop_max_rounds(self):
        """Summary appends tools_exec_budget when set_tool_loop_max_rounds was called."""
        tracker = TokenBudgetTracker(
            token_handler=TokenHandler(),
            context_window_size=10_000,
            max_response_tokens=100,
            max_tool_tokens=10_000,
            round_cap_fraction=0.6,
        )
        tracker.charge(TokenCategory.TOOL_DEFINITIONS, 500)
        tracker.set_tool_loop_max_rounds(20)
        round_index = 3
        horizon = 20 - round_index
        expected = tracker.tools_round_execution_budget(horizon)
        text = tracker.summary(round_index)
        assert f"tools_exec_budget={expected}" in text
        assert f"tool_rounds_left={horizon}" in text
        assert tracker.last_tools_exec_budget == expected
        assert tracker.last_tool_rounds_left == horizon

    def test_summary_omits_exec_budget_without_set_tool_loop_max_rounds(self):
        """Summary does not mention tools_exec_budget until max rounds is registered."""
        tracker = TokenBudgetTracker(
            token_handler=TokenHandler(),
            context_window_size=1000,
            max_response_tokens=100,
            max_tool_tokens=200,
            round_cap_fraction=DEFAULT_TOOL_ROUND_CAP_FRACTION,
        )
        assert "tools_exec_budget" not in tracker.summary(1)
        assert tracker.last_tools_exec_budget is None
        assert tracker.last_tool_rounds_left is None
