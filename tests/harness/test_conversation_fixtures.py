"""Tests for conversation fixture integrity.

Validates that every canned conversation has the correct structure,
role sequencing, and approximate token size for its intended scenario.
"""

import pytest

from tests.harness.conversation_fixtures import (
    CONVERSATION_FIXTURES,
    bloated_tool_result,
    medium_conversation,
    recent_tool_conversation,
    sacred_first_message,
    short_conversation,
)

VALID_ROLES = {"user", "assistant", "tool", "system"}


class TestFixtureStructure:
    """Every fixture must return well-formed message dicts."""

    @staticmethod
    @pytest.mark.parametrize("name", list(CONVERSATION_FIXTURES.keys()))
    def test_fixture_returns_list_of_dicts(name):
        """Each fixture should return a non-empty list of dicts."""
        msgs = CONVERSATION_FIXTURES[name]()
        assert isinstance(msgs, list)
        assert len(msgs) > 0
        for msg in msgs:
            assert isinstance(msg, dict)

    @staticmethod
    @pytest.mark.parametrize("name", list(CONVERSATION_FIXTURES.keys()))
    def test_messages_have_role_and_content(name):
        """Every message must have ``role`` and ``content`` keys."""
        for msg in CONVERSATION_FIXTURES[name]():
            assert "role" in msg, f"Missing 'role' in {msg}"
            assert "content" in msg, f"Missing 'content' in {msg}"

    @staticmethod
    @pytest.mark.parametrize("name", list(CONVERSATION_FIXTURES.keys()))
    def test_roles_are_valid(name):
        """All roles must be one of the standard set."""
        for msg in CONVERSATION_FIXTURES[name]():
            assert msg["role"] in VALID_ROLES, f"Bad role: {msg['role']}"


class TestToolMessagePairing:
    """Tool messages must be preceded by an assistant message with tool_calls."""

    @staticmethod
    @pytest.mark.parametrize("name", list(CONVERSATION_FIXTURES.keys()))
    def test_tool_results_have_matching_call(name):
        """Every tool message must reference a tool_call_id from a prior assistant."""
        msgs = CONVERSATION_FIXTURES[name]()
        emitted_ids: set[str] = set()
        for msg in msgs:
            if msg["role"] == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    emitted_ids.add(tc["id"])
            if msg["role"] == "tool":
                assert "tool_call_id" in msg, "Tool message missing tool_call_id"
                assert msg["tool_call_id"] in emitted_ids, (
                    f"tool_call_id {msg['tool_call_id']} not found in prior "
                    f"assistant tool_calls"
                )


class TestFixtureSizes:
    """Approximate size checks for each scenario."""

    @staticmethod
    def _total_chars(msgs: list[dict]) -> int:
        return sum(len(str(m.get("content", ""))) for m in msgs)

    def test_short_is_small(self):
        """Short conversation should be under 2k chars."""
        assert self._total_chars(short_conversation()) < 2_000

    def test_medium_is_moderate(self):
        """Medium conversation should be between 2k and 20k chars."""
        total = self._total_chars(medium_conversation())
        assert 2_000 < total < 20_000

    def test_bloated_is_large(self):
        """Bloated fixture should exceed 40k chars (oversized tool result)."""
        assert self._total_chars(bloated_tool_result()) > 40_000

    def test_sacred_starts_with_important_context(self):
        """The first message in the sacred fixture must contain constraints."""
        msgs = sacred_first_message()
        first = msgs[0]["content"]
        assert "FIPS" in first
        assert "PCI-DSS" in first
        assert msgs[0]["role"] == "user"

    def test_sacred_has_many_turns(self):
        """Sacred fixture should have at least 20 messages."""
        assert len(sacred_first_message()) >= 20

    def test_recent_tool_pair_near_end(self):
        """The tool call/result pair must be in the last 5 messages."""
        msgs = recent_tool_conversation()
        tool_indices = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
        assert tool_indices, "No tool messages found"
        last_tool_idx = tool_indices[-1]
        assert last_tool_idx >= len(msgs) - 5, (
            f"Tool result at index {last_tool_idx} is not near the end "
            f"(total messages: {len(msgs)})"
        )
