"""Unit tests for tools module."""

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel

from ols.src.tools.tools import (
    DO_NOT_RETRY_REMINDER,
    _extract_text_from_tool_output,
    enforce_tool_token_budget,
    execute_tool_call,
    execute_tool_calls,
    get_tool_by_name,
)

_LARGE_TOKEN_BUDGET = 100_000


class FakeSchema(BaseModel):
    """Fake schema for FakeTool."""


def _make_fake_coroutine(
    name: str, delay: float = 0.0, should_fail: bool = False
) -> Any:
    """Create async coroutine returning (content, artifact) like MCP adapters."""

    async def _coro(**kwargs: Any) -> tuple:
        if delay > 0:
            await asyncio.sleep(delay)
        if should_fail:
            raise Exception("Tool execution failed")
        return f"fake_output_from_{name}", {}

    return _coro


class FakeTool(StructuredTool):
    """Mock tool class that inherits from StructuredTool."""

    def __init__(self, name: str, delay: float = 0.0, should_fail: bool = False):
        """Initialize the tool with a name."""
        super().__init__(
            name=name,
            description=f"Fake tool {name}",
            func=lambda **kwargs: f"fake_output_from_{name}",
            coroutine=_make_fake_coroutine(name, delay, should_fail),
            args_schema=FakeSchema,
        )


class ContentBlockTool(StructuredTool):
    """Mock tool that returns content blocks (langchain-mcp-adapters>=0.2.0 format)."""

    def __init__(self, name: str):
        """Initialize the tool."""

        async def _content_block_coro(**kwargs: Any) -> Any:
            content = [
                {
                    "type": "text",
                    "text": "pod1 Running\npod2 Running",
                    "id": "lc_test_id",
                }
            ]
            return content, {}

        super().__init__(
            name=name,
            description=f"Content block tool {name}",
            func=lambda **kwargs: "sync output",
            coroutine=_content_block_coro,
            args_schema=FakeSchema,
        )


def test_get_tool_by_name():
    """Test get_tool_by_name function."""
    fake_tool_name = "fake_tool"
    fake_tools = [FakeTool(name="fake_tool")]
    fake_tools_duplicite = [FakeTool(name="fake_tool"), FakeTool(name="fake_tool")]

    tool = get_tool_by_name(fake_tool_name, fake_tools)
    assert tool.name == fake_tool_name

    with pytest.raises(ValueError, match=r"Tool 'non_existent_tool' not found\."):
        get_tool_by_name("non_existent_tool", fake_tools)

    with pytest.raises(
        ValueError, match=r"Multiple tools found with name 'fake_tool'\."
    ):
        get_tool_by_name(fake_tool_name, fake_tools_duplicite)


@pytest.mark.asyncio
async def test_execute_tool_call():
    """Test execute_tool_call function."""
    fake_tool_name = "fake_tool"
    fake_tool_args = {"arg1": "value1"}
    fake_tools = [FakeTool(name="fake_tool")]

    with patch(
        "ols.src.tools.tools.get_tool_by_name", return_value=FakeTool(fake_tool_name)
    ):
        output, was_truncated, structured = await execute_tool_call(
            fake_tool_name,
            fake_tool_args,
            fake_tools,
            tools_token_budget=_LARGE_TOKEN_BUDGET,
        )
        assert output == "fake_output_from_fake_tool"
        assert was_truncated is False
        assert structured is None

    with patch(
        "ols.src.tools.tools.get_tool_by_name", side_effect=Exception("Tool error")
    ):
        with pytest.raises(Exception, match="Tool error"):
            await execute_tool_call(
                fake_tool_name,
                fake_tool_args,
                fake_tools,
                tools_token_budget=_LARGE_TOKEN_BUDGET,
            )


@pytest.mark.asyncio
async def test_execute_tool_calls_empty():
    """Test execute_tool_calls with empty tool calls list."""
    tool_messages = await execute_tool_calls(
        [], [], tools_token_budget=_LARGE_TOKEN_BUDGET
    )
    assert tool_messages == []


@pytest.mark.asyncio
async def test_execute_tool_calls_parallel_execution():
    """Test that tool calls are executed in parallel, not sequentially."""
    # Create tools with delays to test parallel execution
    fake_tools = [
        FakeTool(name="tool1", delay=0.1),
        FakeTool(name="tool2", delay=0.1),
        FakeTool(name="tool3", delay=0.1),
    ]

    tool_calls = [
        {"name": "tool1", "args": {}, "id": "call_1"},
        {"name": "tool2", "args": {}, "id": "call_2"},
        {"name": "tool3", "args": {}, "id": "call_3"},
    ]

    start_time = time.time()
    tool_messages = await execute_tool_calls(
        tool_calls, fake_tools, tools_token_budget=_LARGE_TOKEN_BUDGET
    )
    end_time = time.time()

    # If executed in parallel, total time should be close to 0.1s (the delay)
    # If executed sequentially, it would be close to 0.3s (3 * 0.1s)
    execution_time = end_time - start_time
    assert (
        execution_time < 0.25
    ), f"Execution took {execution_time}s, expected < 0.25s for parallel execution"

    # Check that all tools were executed
    assert len(tool_messages) == 3
    assert tool_messages[0].content == "fake_output_from_tool1"
    assert tool_messages[1].content == "fake_output_from_tool2"
    assert tool_messages[2].content == "fake_output_from_tool3"


@pytest.mark.asyncio
async def test_execute_tool_calls_with_missing_tool_name():
    """Test execute_tool_calls with missing tool name."""
    fake_tools = [FakeTool(name="fake_tool")]

    tool_calls = [
        {"name": None, "args": {}, "id": "call_1"},  # Missing tool name
        {"name": "fake_tool", "args": {}, "id": "call_2"},  # Valid tool call
    ]

    with patch(
        "ols.src.tools.tools.execute_tool_call",
        return_value=("fake_output", False, None),
    ):
        tool_messages = await execute_tool_calls(
            tool_calls, fake_tools, tools_token_budget=_LARGE_TOKEN_BUDGET
        )
        assert len(tool_messages) == 2
        assert (
            tool_messages[0].content
            == "Error: Tool name is missing from tool call. Do not retry this exact tool call."
        )
        assert tool_messages[0].status == "error"
        assert tool_messages[0].tool_call_id == "call_1"
        assert tool_messages[0].additional_kwargs.get("truncated") is False
        assert tool_messages[1].content == "fake_output"
        assert tool_messages[1].status == "success"
        assert tool_messages[1].tool_call_id == "call_2"
        assert tool_messages[1].additional_kwargs.get("truncated") is False


@pytest.mark.asyncio
async def test_execute_tool_calls_mixed_success_and_failure():
    """Test execute_tool_calls with a mix of successful and failing tools."""
    fake_tools = [
        FakeTool(name="success_tool"),
        FakeTool(name="fail_tool", should_fail=True),
    ]

    tool_calls = [
        {"name": "success_tool", "args": {}, "id": "call_1"},
        {"name": "fail_tool", "args": {}, "id": "call_2"},
        {"name": "nonexistent_tool", "args": {}, "id": "call_3"},
    ]

    tool_messages = await execute_tool_calls(
        tool_calls, fake_tools, tools_token_budget=_LARGE_TOKEN_BUDGET
    )

    assert len(tool_messages) == 3

    # First call should succeed
    assert tool_messages[0].status == "success"
    assert tool_messages[0].content == "fake_output_from_success_tool"

    # Second call should fail due to tool raising exception
    assert tool_messages[1].status == "error"
    assert "execution failed after 1 attempt(s)" in tool_messages[1].content
    assert "Tool execution failed" in tool_messages[1].content
    assert DO_NOT_RETRY_REMINDER in tool_messages[1].content

    # Third call should fail due to nonexistent tool
    assert tool_messages[2].status == "error"
    assert "Tool 'nonexistent_tool' not found" in tool_messages[2].content
    assert "execution failed after 1 attempt(s)" in tool_messages[2].content
    assert DO_NOT_RETRY_REMINDER in tool_messages[2].content


@pytest.mark.asyncio
async def test_execute_tool_calls_retries_transient_error_then_succeeds():
    """Retry transient error and succeed on the next attempt."""
    call_count = 0

    async def _flaky_coro(**kwargs: Any) -> tuple:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("connection reset by peer")
        return "ok_after_retry", {}

    tool = StructuredTool(
        name="flaky_tool",
        description="Flaky tool",
        func=lambda **kwargs: "sync",
        coroutine=_flaky_coro,
        args_schema=FakeSchema,
    )

    tool_messages = await execute_tool_calls(
        [{"name": "flaky_tool", "args": {}, "id": "call_1"}],
        [tool],
        tools_token_budget=_LARGE_TOKEN_BUDGET,
    )

    assert len(tool_messages) == 1
    assert call_count == 2
    assert tool_messages[0].status == "success"
    assert tool_messages[0].content == "ok_after_retry"


@pytest.mark.asyncio
async def test_execute_tool_calls_retry_exhausted_returns_no_retry_reminder():
    """Return error with retry-exhausted message and no-retry reminder."""
    call_count = 0

    async def _always_timeout(**kwargs: Any) -> tuple:
        nonlocal call_count
        call_count += 1
        raise ConnectionError("timeout during MCP call")

    tool = StructuredTool(
        name="timeout_tool",
        description="Always fails with retryable error",
        func=lambda **kwargs: "sync",
        coroutine=_always_timeout,
        args_schema=FakeSchema,
    )

    tool_messages = await execute_tool_calls(
        [{"name": "timeout_tool", "args": {}, "id": "call_1"}],
        [tool],
        tools_token_budget=_LARGE_TOKEN_BUDGET,
    )

    assert len(tool_messages) == 1
    assert call_count == 3
    assert tool_messages[0].status == "error"
    assert "execution failed after 3 attempt(s)" in tool_messages[0].content
    assert DO_NOT_RETRY_REMINDER in tool_messages[0].content


@pytest.mark.asyncio
async def test_execute_tool_calls_non_retryable_error_does_not_retry():
    """Do not retry non-transient exceptions."""
    call_count = 0

    async def _non_retryable_failure(**kwargs: Any) -> tuple:
        nonlocal call_count
        call_count += 1
        raise ValueError("invalid tool arguments")

    tool = StructuredTool(
        name="bad_tool",
        description="Always fails with non-retryable error",
        func=lambda **kwargs: "sync",
        coroutine=_non_retryable_failure,
        args_schema=FakeSchema,
    )

    tool_messages = await execute_tool_calls(
        [{"name": "bad_tool", "args": {}, "id": "call_1"}],
        [tool],
        tools_token_budget=_LARGE_TOKEN_BUDGET,
    )

    assert len(tool_messages) == 1
    assert call_count == 1
    assert tool_messages[0].status == "error"
    assert "execution failed after 1 attempt(s)" in tool_messages[0].content
    assert DO_NOT_RETRY_REMINDER in tool_messages[0].content


@pytest.mark.asyncio
async def test_execute_tool_calls_preserves_tool_call_order():
    """Test that tool messages are returned in the same order as tool calls."""
    fake_tools = [
        FakeTool(name="tool_a"),
        FakeTool(name="tool_b"),
        FakeTool(name="tool_c"),
    ]

    tool_calls = [
        {"name": "tool_c", "args": {}, "id": "call_c"},
        {"name": "tool_a", "args": {}, "id": "call_a"},
        {"name": "tool_b", "args": {}, "id": "call_b"},
    ]

    tool_messages = await execute_tool_calls(
        tool_calls, fake_tools, tools_token_budget=_LARGE_TOKEN_BUDGET
    )

    assert len(tool_messages) == 3
    # Order should match input order, not alphabetical
    assert tool_messages[0].tool_call_id == "call_c"
    assert tool_messages[0].content == "fake_output_from_tool_c"
    assert tool_messages[1].tool_call_id == "call_a"
    assert tool_messages[1].content == "fake_output_from_tool_a"
    assert tool_messages[2].tool_call_id == "call_b"
    assert tool_messages[2].content == "fake_output_from_tool_b"


class LargeOutputTool(StructuredTool):
    """Tool that returns a large output for testing truncation."""

    def __init__(self, name: str, output_size: int = 100):
        """Initialize with configurable output size."""

        async def _large_coro(**kwargs: Any) -> tuple:
            return "word " * output_size, {}

        super().__init__(
            name=name,
            description=f"Large output tool {name}",
            func=lambda **kwargs: "word " * output_size,
            coroutine=_large_coro,
            args_schema=FakeSchema,
        )


@pytest.mark.asyncio
async def test_execute_tool_call_with_truncation():
    """Test that large tool outputs are truncated."""
    large_tool = LargeOutputTool(name="large_tool", output_size=5000)
    fake_tools = [large_tool]

    output, was_truncated, structured = await execute_tool_call(
        "large_tool", {}, fake_tools, tools_token_budget=100
    )

    assert was_truncated is True
    assert structured is None
    assert "[OUTPUT TRUNCATED" in output
    assert "Please ask a more specific question" in output


@pytest.mark.asyncio
async def test_execute_tool_call_no_truncation_small_output():
    """Test that small tool outputs are not truncated."""
    small_tool = LargeOutputTool(name="small_tool", output_size=10)
    fake_tools = [small_tool]

    output, was_truncated, structured = await execute_tool_call(
        "small_tool", {}, fake_tools, tools_token_budget=1000
    )

    assert was_truncated is False
    assert structured is None
    assert "[OUTPUT TRUNCATED" not in output


@pytest.mark.asyncio
async def test_execute_tool_calls_truncation_in_additional_kwargs():
    """Test that truncation flag is stored in additional_kwargs."""
    large_tool = LargeOutputTool(name="large_tool", output_size=5000)
    small_tool = LargeOutputTool(name="small_tool", output_size=10)
    fake_tools = [large_tool, small_tool]

    tool_calls = [
        {"name": "large_tool", "args": {}, "id": "call_large"},
        {"name": "small_tool", "args": {}, "id": "call_small"},
    ]

    tool_messages = await execute_tool_calls(
        tool_calls, fake_tools, tools_token_budget=100
    )

    assert len(tool_messages) == 2

    # Large tool should be truncated
    assert tool_messages[0].additional_kwargs.get("truncated") is True
    assert "[OUTPUT TRUNCATED" in tool_messages[0].content

    # Small tool should not be truncated
    assert tool_messages[1].additional_kwargs.get("truncated") is False
    assert "[OUTPUT TRUNCATED" not in tool_messages[1].content


@pytest.mark.asyncio
async def test_execute_tool_calls_custom_token_budget():
    """Test that tools_token_budget is respected."""
    tool = LargeOutputTool(name="test_tool", output_size=500)
    fake_tools = [tool]

    tool_calls = [{"name": "test_tool", "args": {}, "id": "call_1"}]

    # With high limit - no truncation
    messages_high_limit = await execute_tool_calls(
        tool_calls, fake_tools, tools_token_budget=10000
    )
    assert messages_high_limit[0].additional_kwargs.get("truncated") is False

    # With low limit - truncation
    messages_low_limit = await execute_tool_calls(
        tool_calls, fake_tools, tools_token_budget=50
    )
    assert messages_low_limit[0].additional_kwargs.get("truncated") is True


def test_extract_text_from_tool_output_string():
    """Test _extract_text_from_tool_output with a plain string."""
    assert _extract_text_from_tool_output(
        "hello", tools_token_budget=_LARGE_TOKEN_BUDGET
    ) == ("hello", False)


def test_extract_text_from_tool_output_content_blocks():
    """Test _extract_text_from_tool_output with LC standard content blocks."""
    blocks = [
        {"type": "text", "text": "pod1 Running", "id": "lc_1"},
        {"type": "text", "text": "pod2 Running", "id": "lc_2"},
    ]
    text, truncated = _extract_text_from_tool_output(
        blocks, tools_token_budget=_LARGE_TOKEN_BUDGET
    )
    assert text == "pod1 Running\npod2 Running"
    assert truncated is False


def test_extract_text_from_tool_output_single_content_block():
    """Test _extract_text_from_tool_output with a single content block."""
    blocks = [{"type": "text", "text": "some output", "id": "lc_1"}]
    assert _extract_text_from_tool_output(
        blocks, tools_token_budget=_LARGE_TOKEN_BUDGET
    ) == ("some output", False)


def test_extract_text_from_tool_output_mixed_list():
    """Test _extract_text_from_tool_output with mixed list elements."""
    blocks = [
        {"type": "text", "text": "text content"},
        "plain string",
    ]
    text, truncated = _extract_text_from_tool_output(
        blocks, tools_token_budget=_LARGE_TOKEN_BUDGET
    )
    assert text == "text content\nplain string"
    assert truncated is False


def test_extract_text_from_tool_output_non_string_non_list():
    """Test _extract_text_from_tool_output with unexpected types."""
    assert _extract_text_from_tool_output(
        42, tools_token_budget=_LARGE_TOKEN_BUDGET
    ) == ("42", False)
    assert _extract_text_from_tool_output(
        None, tools_token_budget=_LARGE_TOKEN_BUDGET
    ) == ("None", False)


def test_extract_text_from_tool_output_empty_list():
    """Test _extract_text_from_tool_output with empty list."""
    assert _extract_text_from_tool_output(
        [], tools_token_budget=_LARGE_TOKEN_BUDGET
    ) == ("", False)


def test_extract_text_from_tool_output_string_truncation():
    """Test that a long string is truncated at the last newline boundary."""
    long_output = "line\n" * 5000
    text, truncated = _extract_text_from_tool_output(long_output, tools_token_budget=10)
    assert truncated is True
    assert len(text) < len(long_output)
    assert not text.endswith("\n\n")


def test_extract_text_from_tool_output_list_block_truncation():
    """Test that list blocks are dropped when they exceed the char limit."""
    blocks = [
        {"type": "text", "text": "block " * 200},
        {"type": "text", "text": "block " * 200},
        {"type": "text", "text": "block " * 200},
    ]
    text, truncated = _extract_text_from_tool_output(blocks, tools_token_budget=10)
    assert truncated is True
    assert len(text) < sum(len(b["text"]) for b in blocks)


@pytest.mark.asyncio
async def test_execute_tool_call_with_content_blocks():
    """Test execute_tool_call handles content block output from MCP adapters 0.2+."""
    content_block_tool = ContentBlockTool(name="content_tool")
    fake_tools = [content_block_tool]

    output, was_truncated, structured = await execute_tool_call(
        "content_tool", {}, fake_tools, tools_token_budget=_LARGE_TOKEN_BUDGET
    )
    assert output == "pod1 Running\npod2 Running"
    assert was_truncated is False
    assert structured is None


class ArtifactTool(StructuredTool):
    """Mock tool that returns (content, artifact) tuple like langchain-mcp-adapters 0.2+."""

    def __init__(self, name: str):
        """Initialize the tool."""

        async def _artifact_coro(**kwargs: Any) -> Any:
            content = [{"type": "text", "text": "Pod utilization summary"}]
            artifact = {
                "structured_content": {
                    "pods": [
                        {"name": "pod1", "cpu": 45, "memory": 62},
                        {"name": "pod2", "cpu": 12, "memory": 34},
                    ],
                    "summary": {"totalPods": 2, "avgCpu": 28, "avgMemory": 48},
                }
            }
            return content, artifact

        super().__init__(
            name=name,
            description=f"Artifact tool {name}",
            func=lambda **kwargs: "sync output",
            coroutine=_artifact_coro,
            response_format="content_and_artifact",
            args_schema=FakeSchema,
        )


@pytest.mark.asyncio
async def test_execute_tool_call_with_structured_content():
    """Test execute_tool_call captures structured_content from artifact tuple."""
    artifact_tool = ArtifactTool(name="artifact_tool")
    fake_tools = [artifact_tool]

    output, was_truncated, structured = await execute_tool_call(
        "artifact_tool", {}, fake_tools, tools_token_budget=_LARGE_TOKEN_BUDGET
    )
    assert output == "Pod utilization summary"
    assert was_truncated is False
    assert structured is not None
    assert "pods" in structured
    assert len(structured["pods"]) == 2
    assert structured["summary"]["totalPods"] == 2


@pytest.mark.asyncio
async def test_execute_tool_call_structured_content_propagated_to_tool_message():
    """Test structured_content is forwarded in ToolMessage additional_kwargs."""
    artifact_tool = ArtifactTool(name="artifact_tool")
    fake_tools = [artifact_tool]

    tool_calls = [{"name": "artifact_tool", "args": {}, "id": "call_art"}]
    tool_messages = await execute_tool_calls(
        tool_calls, fake_tools, tools_token_budget=_LARGE_TOKEN_BUDGET
    )

    assert len(tool_messages) == 1
    msg = tool_messages[0]
    assert msg.status == "success"
    assert msg.content == "Pod utilization summary"
    assert msg.additional_kwargs.get("truncated") is False
    structured = msg.additional_kwargs.get("structured_content")
    assert structured is not None
    assert structured["summary"]["totalPods"] == 2


@pytest.mark.asyncio
async def test_execute_tool_call_artifact_without_structured_content():
    """Test that artifact dicts without structured_content key return None."""

    async def _coro(**kwargs: Any) -> Any:
        return [{"type": "text", "text": "plain result"}], {"other_key": "value"}

    tool = StructuredTool(
        name="no_sc_tool",
        description="Tool with artifact but no structured_content",
        func=lambda **kwargs: "sync",
        coroutine=_coro,
        args_schema=FakeSchema,
    )

    output, _was_truncated, structured = await execute_tool_call(
        "no_sc_tool", {}, [tool], tools_token_budget=_LARGE_TOKEN_BUDGET
    )
    assert output == "plain result"
    assert structured is None


def _make_tool_message(content: str, tool_call_id: str = "id") -> ToolMessage:
    """Create a ToolMessage for testing."""
    return ToolMessage(
        content=content,
        status="success",
        tool_call_id=tool_call_id,
        additional_kwargs={"truncated": False},
    )


def test_enforce_tool_token_budget_empty_list():
    """Test that an empty list is returned unchanged."""
    assert enforce_tool_token_budget([], 100) == []


def test_enforce_tool_token_budget_under_budget():
    """Test that small messages pass through without truncation."""
    msgs = [_make_tool_message("short reply", "c1")]
    result = enforce_tool_token_budget(msgs, 10000)
    assert len(result) == 1
    assert result[0].content == "short reply"
    assert result[0].additional_kwargs["truncated"] is False


def test_enforce_tool_token_budget_truncates_longest():
    """Test that only the longest message is truncated when it dominates."""
    long_content = "line\n" * 2000
    short_content = "ok"
    msgs = [
        _make_tool_message(long_content, "c_long"),
        _make_tool_message(short_content, "c_short"),
    ]
    result = enforce_tool_token_budget(msgs, 2500)

    assert result[0].additional_kwargs["truncated"] is True
    assert "[OUTPUT TRUNCATED" in result[0].content
    assert len(result[0].content) < len(long_content)

    assert result[1].content == short_content
    assert result[1].additional_kwargs["truncated"] is False


def test_enforce_tool_token_budget_proportional_truncation():
    """Test proportional truncation when no single message dominates."""
    content_a = "word " * 500
    content_b = "data " * 500
    msgs = [
        _make_tool_message(content_a, "c_a"),
        _make_tool_message(content_b, "c_b"),
    ]
    result = enforce_tool_token_budget(msgs, 20)

    assert result[0].additional_kwargs["truncated"] is True
    assert result[1].additional_kwargs["truncated"] is True
    assert "[OUTPUT TRUNCATED" in result[0].content
    assert "[OUTPUT TRUNCATED" in result[1].content


def test_enforce_tool_token_budget_preserves_metadata():
    """Test that tool_call_id and status are preserved after truncation."""
    msgs = [_make_tool_message("word " * 1000, "my_call_id")]
    msgs[0].status = "success"
    result = enforce_tool_token_budget(msgs, 20)

    assert result[0].tool_call_id == "my_call_id"
    assert result[0].status == "success"
