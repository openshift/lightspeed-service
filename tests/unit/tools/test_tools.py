"""Unit tests for tools module."""

import asyncio
import logging
import time
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel

from ols.src.tools import tools as tools_module
from ols.src.tools.tools import (
    ToolResultEvent,
    _extract_text_from_tool_output,
    _is_transient_tool_error,
    enforce_tool_token_budget,
    execute_tool_call,
    execute_tool_calls_stream,
    get_tool_by_name,
)
from ols.utils.token_handler import TokenHandler

_LARGE_TOKEN_BUDGET = 100_000


class FakeSchema(BaseModel):
    """Fake schema for fake tools."""


def _make_fake_coroutine(
    name: str, delay: float = 0.0, should_fail: bool = False
) -> Any:
    """Create async coroutine used by FakeTool."""

    async def _coro(**kwargs: Any) -> str:
        if delay > 0:
            await asyncio.sleep(delay)
        if should_fail:
            raise Exception("Tool execution failed")
        return f"fake_output_from_{name}"

    return _coro


class FakeTool(StructuredTool):
    """Mock tool class that inherits from StructuredTool."""

    def __init__(
        self,
        name: str,
        delay: float = 0.0,
        should_fail: bool = False,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize the tool with a name."""
        super().__init__(
            name=name,
            description=f"Fake tool {name}",
            func=lambda **kwargs: f"fake_output_from_{name}",
            coroutine=_make_fake_coroutine(name, delay, should_fail),
            args_schema=FakeSchema,
            metadata=metadata,
        )


class ContentBlockTool(StructuredTool):
    """Mock tool returning content blocks."""

    def __init__(self, name: str):
        """Initialize the tool."""

        async def _content_block_coro(**kwargs: Any) -> Any:
            return [
                {
                    "type": "text",
                    "text": "pod1 Running\npod2 Running",
                    "id": "lc_test_id",
                }
            ]

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


class LargeOutputTool(StructuredTool):
    """Tool returning large output for truncation tests."""

    def __init__(self, name: str, output_size: int = 100):
        """Initialize with configurable output size."""

        async def _large_coro(**kwargs: Any) -> str:
            return "word " * output_size

        super().__init__(
            name=name,
            description=f"Large output tool {name}",
            func=lambda **kwargs: "word " * output_size,
            coroutine=_large_coro,
            args_schema=FakeSchema,
        )


class TupleOutputTool(StructuredTool):
    """Tool that returns (content, artifact) tuple."""

    def __init__(self, name: str):
        """Initialize tuple-output mock tool."""

        async def _tuple_coro(**kwargs: Any) -> tuple[Any, Any]:
            return ([{"type": "text", "text": "tuple content"}], {"artifact": 1})

        super().__init__(
            name=name,
            description=f"Tuple output tool {name}",
            func=lambda **kwargs: "unused",
            coroutine=_tuple_coro,
            args_schema=FakeSchema,
        )


class NonDictStructuredContentTool(StructuredTool):
    """Tool whose artifact reports structured_content that is not a dict."""

    def __init__(self, name: str):
        """Initialize mock tool."""

        async def _coro(**kwargs: Any) -> tuple[Any, Any]:
            return ("ok", {"structured_content": ["x"]})

        super().__init__(
            name=name,
            description=f"Non-dict structured_content tool {name}",
            func=lambda **kwargs: "unused",
            coroutine=_coro,
            args_schema=FakeSchema,
        )


async def _collect_tool_messages(
    tool_calls: list[tuple[str, dict[str, Any], StructuredTool]],
    tools_token_budget: int = 4000,
) -> list[ToolMessage]:
    """Execute tool stream and collect only tool_result messages."""
    return [
        event.data
        async for event in execute_tool_calls_stream(
            tool_calls, tools_token_budget=tools_token_budget, streaming=False
        )
        if event.event == "tool_result"
    ]


@pytest.mark.asyncio
async def test_execute_tool_call_success() -> None:
    """Test execute_tool_call success path."""
    status, output, was_truncated, structured_content = await execute_tool_call(
        FakeTool("fake_tool"), {"a": 1}, _LARGE_TOKEN_BUDGET
    )
    assert output == "fake_output_from_fake_tool"
    assert status == "success"
    assert was_truncated is False
    assert structured_content is None


@pytest.mark.asyncio
async def test_execute_tool_call_failure_raises() -> None:
    """Test execute_tool_call raises tool exception."""
    with pytest.raises(Exception, match="Tool execution failed"):
        await execute_tool_call(
            FakeTool("bad", should_fail=True), {}, _LARGE_TOKEN_BUDGET
        )


@pytest.mark.asyncio
async def test_execute_tool_calls_stream_empty() -> None:
    """Test execute_tool_calls_stream with empty list."""
    events = [
        event
        async for event in execute_tool_calls_stream(
            [], tools_token_budget=_LARGE_TOKEN_BUDGET, streaming=False
        )
    ]
    assert events == []


@pytest.mark.asyncio
async def test_execute_tool_calls_stream_parallel_execution() -> None:
    """Test that tool streams execute in parallel."""
    tool_calls = [
        ("call_1", {}, FakeTool("tool1", delay=0.1)),
        ("call_2", {}, FakeTool("tool2", delay=0.1)),
        ("call_3", {}, FakeTool("tool3", delay=0.1)),
    ]
    start_time = time.time()
    tool_messages = await _collect_tool_messages(tool_calls)
    execution_time = time.time() - start_time
    assert execution_time < 0.25
    assert len(tool_messages) == 3
    outputs = {m.content for m in tool_messages}
    assert outputs == {
        "fake_output_from_tool1",
        "fake_output_from_tool2",
        "fake_output_from_tool3",
    }


@pytest.mark.asyncio
async def test_execute_tool_calls_mixed_success_and_failure() -> None:
    """Test mixed successful and failing tool calls."""
    tool_calls = [
        ("call_1", {}, FakeTool("success_tool")),
        ("call_2", {}, FakeTool("fail_tool", should_fail=True)),
    ]
    tool_messages = await _collect_tool_messages(tool_calls)
    assert len(tool_messages) == 2
    by_id = {m.tool_call_id: m for m in tool_messages}
    assert by_id["call_1"].status == "success"
    assert by_id["call_1"].content == "fake_output_from_success_tool"
    assert by_id["call_2"].status == "error"
    assert "Tool 'fail_tool' failed:" in by_id["call_2"].content
    assert by_id["call_2"].additional_kwargs["truncated"] is False


@pytest.mark.asyncio
async def test_execute_tool_call_with_truncation() -> None:
    """Test large tool outputs are truncated."""
    status, output, was_truncated, _ = await execute_tool_call(
        LargeOutputTool(name="large_tool", output_size=5000),
        {},
        tools_token_budget=100,
    )
    assert status == "success"
    assert was_truncated is True
    assert "[OUTPUT TRUNCATED" in output


@pytest.mark.asyncio
async def test_execute_tool_call_no_truncation_small_output() -> None:
    """Test small outputs are not truncated."""
    status, output, was_truncated, _ = await execute_tool_call(
        LargeOutputTool(name="small_tool", output_size=10),
        {},
        tools_token_budget=1000,
    )
    assert status == "success"
    assert was_truncated is False
    assert "[OUTPUT TRUNCATED" not in output


@pytest.mark.asyncio
async def test_execute_tool_calls_truncation_in_additional_kwargs() -> None:
    """Test truncation flag is stored in additional_kwargs."""
    tool_calls = [
        ("call_large", {}, LargeOutputTool(name="large_tool", output_size=5000)),
        ("call_small", {}, LargeOutputTool(name="small_tool", output_size=10)),
    ]
    tool_messages = await _collect_tool_messages(tool_calls, tools_token_budget=100)
    by_id = {m.tool_call_id: m for m in tool_messages}
    assert by_id["call_large"].additional_kwargs.get("truncated") is True
    assert "[OUTPUT TRUNCATED" in by_id["call_large"].content
    assert by_id["call_small"].additional_kwargs.get("truncated") is False
    assert "[OUTPUT TRUNCATED" not in by_id["call_small"].content


def test_extract_text_from_tool_output_string():
    """Test _extract_text_from_tool_output with a plain string."""
    assert _extract_text_from_tool_output(
        "hello", tools_token_budget=_LARGE_TOKEN_BUDGET
    ) == ("hello", False)


def test_extract_text_from_tool_output_content_blocks() -> None:
    """Test _extract_text_from_tool_output with content blocks."""
    blocks = [
        {"type": "text", "text": "pod1 Running", "id": "lc_1"},
        {"type": "text", "text": "pod2 Running", "id": "lc_2"},
    ]
    text, truncated = _extract_text_from_tool_output(
        blocks, tools_token_budget=_LARGE_TOKEN_BUDGET
    )
    assert text == "pod1 Running\npod2 Running"
    assert truncated is False


def test_extract_text_from_tool_output_single_content_block() -> None:
    """Test _extract_text_from_tool_output with one content block."""
    blocks = [{"type": "text", "text": "some output", "id": "lc_1"}]
    assert _extract_text_from_tool_output(
        blocks, tools_token_budget=_LARGE_TOKEN_BUDGET
    ) == ("some output", False)


def test_extract_text_from_tool_output_mixed_list() -> None:
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


def test_extract_text_from_tool_output_non_string_non_list() -> None:
    """Test _extract_text_from_tool_output with unexpected types."""
    assert _extract_text_from_tool_output(
        42, tools_token_budget=_LARGE_TOKEN_BUDGET
    ) == ("42", False)
    assert _extract_text_from_tool_output(
        None, tools_token_budget=_LARGE_TOKEN_BUDGET
    ) == ("None", False)


def test_extract_text_from_tool_output_empty_list() -> None:
    """Test _extract_text_from_tool_output with empty list."""
    assert _extract_text_from_tool_output(
        [], tools_token_budget=_LARGE_TOKEN_BUDGET
    ) == ("", False)


def test_extract_text_first_block_exceeds_budget_keeps_prefix(caplog):
    """First list block alone exceeds char cap: prefix is kept, then truncation notice."""
    caplog.set_level(logging.DEBUG)
    huge = "x" * 10_000
    text, truncated = _extract_text_from_tool_output(
        [{"type": "text", "text": huge}], tools_token_budget=10
    )
    assert truncated is True
    assert "first block exceeds char budget" in caplog.text
    assert "pre-truncating first block" in caplog.text
    max_chars = 10 * 4
    assert text.endswith(tools_module._TRUNCATION_WARNING)
    assert text.startswith("x" * max_chars)


def test_extract_text_all_empty_text_blocks() -> None:
    """All text blocks empty: no truncation, joined text is whitespace-only."""
    text, truncated = _extract_text_from_tool_output(
        [{"type": "text", "text": ""}, {"type": "text", "text": ""}],
        tools_token_budget=100,
    )
    assert truncated is False
    assert not text.strip()


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


def test_extract_text_from_tool_output_list_with_unknown_item() -> None:
    """Test _extract_text_from_tool_output stringifies unknown list entries."""
    blocks = [{"type": "text", "text": "text content"}, 123]
    text, truncated = _extract_text_from_tool_output(
        blocks, tools_token_budget=_LARGE_TOKEN_BUDGET
    )
    assert text == "text content\n123"
    assert truncated is False


@pytest.mark.asyncio
async def test_execute_tool_call_with_content_blocks() -> None:
    """Test execute_tool_call handles content block output."""
    status, output, was_truncated, _ = await execute_tool_call(
        ContentBlockTool(name="content_tool"),
        {},
        _LARGE_TOKEN_BUDGET,
    )
    assert status == "success"
    assert output == "pod1 Running\npod2 Running"
    assert was_truncated is False


@pytest.mark.asyncio
async def test_execute_tool_call_with_content_and_artifact_tuple() -> None:
    """Test execute_tool_call handles content+artifact tuple output."""
    status, output, was_truncated, structured_content = await execute_tool_call(
        TupleOutputTool(name="tuple_tool"),
        {},
        _LARGE_TOKEN_BUDGET,
    )
    assert status == "success"
    assert output == "tuple content"
    assert was_truncated is False
    assert structured_content is None


@pytest.mark.asyncio
async def test_execute_tool_call_non_dict_structured_content_becomes_none() -> None:
    """Non-dict structured_content in artifact is ignored."""
    status, output, was_truncated, structured_content = await execute_tool_call(
        NonDictStructuredContentTool(name="bad_struct"),
        {},
        _LARGE_TOKEN_BUDGET,
    )
    assert status == "success"
    assert output == "ok"
    assert was_truncated is False
    assert structured_content is None


def test_is_transient_tool_error_branches() -> None:
    """Test retryability classifier for exception-type branches."""
    assert _is_transient_tool_error(TimeoutError("timeout")) is True
    assert _is_transient_tool_error(ConnectionError("reset")) is True
    assert _is_transient_tool_error(OSError("network unreachable")) is True
    assert _is_transient_tool_error(Exception("hard failure")) is False


@pytest.mark.asyncio
async def test_execute_tool_calls_stream_emits_approval_required_then_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test approval flow emits approval_required event before tool_result."""

    async def _approved(*args: object, **kwargs: object) -> str:
        return "approved"

    monkeypatch.setattr(tools_module, "need_validation", lambda **kwargs: True)
    monkeypatch.setattr(tools_module, "get_approval_decision", _approved)

    tool = FakeTool(
        "approved_tool",
        metadata={"annotations": {"readOnlyHint": True, "category": "safe"}},
    )
    events = [
        event
        async for event in execute_tool_calls_stream(
            [("call_approved", {}, tool)],
            tools_token_budget=_LARGE_TOKEN_BUDGET,
            streaming=True,
        )
    ]

    assert len(events) == 2
    assert events[0].event == "approval_required"
    assert events[0].data["tool_name"] == "approved_tool"
    assert events[0].data["tool_annotation"] == {
        "readOnlyHint": True,
        "category": "safe",
    }
    assert isinstance(events[0].data["approval_id"], str)

    assert events[1].event == "tool_result"
    assert events[1].data.tool_call_id == "call_approved"
    assert events[1].data.status == "success"
    assert events[1].data.content == "fake_output_from_approved_tool"


@pytest.mark.asyncio
async def test_execute_tool_calls_stream_rejection_returns_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test rejected approval returns non-retryable synthetic tool result."""

    async def _rejected(*args: object, **kwargs: object) -> str:
        return "rejected"

    async def _must_not_execute(
        *args: object, **kwargs: object
    ) -> tuple[str, str, bool]:
        raise AssertionError(
            "execute_tool_call should not run when approval is rejected"
        )

    monkeypatch.setattr(tools_module, "need_validation", lambda **kwargs: True)
    monkeypatch.setattr(tools_module, "get_approval_decision", _rejected)
    monkeypatch.setattr(tools_module, "execute_tool_call", _must_not_execute)

    events = [
        event
        async for event in execute_tool_calls_stream(
            [("call_rejected", {}, FakeTool("rejected_tool"))],
            tools_token_budget=_LARGE_TOKEN_BUDGET,
            streaming=True,
        )
    ]

    assert len(events) == 2
    assert events[0].event == "approval_required"
    assert events[1].event == "tool_result"
    assert events[1].data.status == "error"
    assert "execution was rejected" in events[1].data.content
    assert "Do not retry this exact tool call." in events[1].data.content


@pytest.mark.asyncio
async def test_execute_tool_calls_stream_timeout_returns_timeout_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test approval timeout returns terminal timeout tool result."""

    async def _timed_out(*args: object, **kwargs: object) -> str:
        return "timeout"

    async def _must_not_execute(
        *args: object, **kwargs: object
    ) -> tuple[str, str, bool]:
        raise AssertionError("execute_tool_call should not run when approval times out")

    monkeypatch.setattr(tools_module, "need_validation", lambda **kwargs: True)
    monkeypatch.setattr(tools_module, "get_approval_decision", _timed_out)
    monkeypatch.setattr(tools_module, "execute_tool_call", _must_not_execute)

    events = [
        event
        async for event in execute_tool_calls_stream(
            [("call_timeout", {}, FakeTool("timeout_tool"))],
            tools_token_budget=_LARGE_TOKEN_BUDGET,
            streaming=True,
        )
    ]

    assert len(events) == 2
    assert events[0].event == "approval_required"
    assert events[1].event == "tool_result"
    assert events[1].data.status == "error"
    assert "approval timed out" in events[1].data.content


@pytest.mark.asyncio
async def test_execute_tool_calls_stream_no_approval_when_not_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test no approval event is emitted when validation is not required."""

    async def _must_not_wait(*args: object, **kwargs: object) -> str:
        raise AssertionError(
            "get_approval_decision should not run when approval is not needed"
        )

    monkeypatch.setattr(tools_module, "need_validation", lambda **kwargs: False)
    monkeypatch.setattr(tools_module, "get_approval_decision", _must_not_wait)

    events = [
        event
        async for event in execute_tool_calls_stream(
            [("call_no_approval", {}, FakeTool("plain_tool"))],
            tools_token_budget=_LARGE_TOKEN_BUDGET,
            streaming=False,
        )
    ]

    assert [event.event for event in events] == ["tool_result"]
    assert events[0].data.content == "fake_output_from_plain_tool"


@pytest.mark.asyncio
async def test_execute_tool_calls_stream_retries_retryable_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test retry loop retries transient error and eventually succeeds."""

    async def _execute_with_one_retry(
        tool: StructuredTool,
        tool_args: dict[str, Any],
        tools_token_budget: int,
    ) -> tuple[str, str, bool, dict | None]:
        count = getattr(_execute_with_one_retry, "count", 0) + 1
        _execute_with_one_retry.count = count
        if count == 1:
            raise TimeoutError("temporary timeout")
        return "success", "retried-ok", False, None

    async def _no_sleep(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(tools_module, "need_validation", lambda **kwargs: False)
    monkeypatch.setattr(tools_module, "execute_tool_call", _execute_with_one_retry)
    monkeypatch.setattr(tools_module.asyncio, "sleep", _no_sleep)

    events = [
        event
        async for event in execute_tool_calls_stream(
            [("call_retry", {}, FakeTool("retry_tool"))],
            tools_token_budget=_LARGE_TOKEN_BUDGET,
            streaming=False,
        )
    ]
    assert len(events) == 1
    assert events[0].event == "tool_result"
    assert events[0].data.content == "retried-ok"


@pytest.mark.asyncio
async def test_execute_tool_calls_stream_cancels_remaining_tasks_on_early_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test stream finalizer cancels unfinished worker tasks on consumer early exit."""

    async def _mixed_execute(tool_call: Any, *args: Any, **kwargs: Any):
        tool_id = tool_call[0]
        if tool_id == "call_fast":
            yield ToolResultEvent(
                data=ToolMessage(
                    content="fast",
                    status="success",
                    tool_call_id="call_fast",
                    additional_kwargs={"truncated": False},
                )
            )
            return
        await asyncio.sleep(1)
        yield ToolResultEvent(
            data=ToolMessage(
                content="slow",
                status="success",
                tool_call_id="call_slow",
                additional_kwargs={"truncated": False},
            )
        )

    monkeypatch.setattr(
        tools_module, "_execute_single_tool_call_stream", _mixed_execute
    )

    gen = execute_tool_calls_stream(
        [
            ("call_fast", {}, FakeTool("tool1")),
            ("call_slow", {}, FakeTool("tool2")),
        ],
        tools_token_budget=_LARGE_TOKEN_BUDGET,
        streaming=False,
    )
    first_event = await gen.__anext__()
    assert first_event.event == "tool_result"
    assert first_event.data.tool_call_id == "call_fast"
    await gen.aclose()


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
    assert enforce_tool_token_budget([], 100, TokenHandler()) == []


def test_enforce_tool_token_budget_under_budget():
    """Test that small messages pass through without truncation."""
    msgs = [_make_tool_message("short reply", "c1")]
    result = enforce_tool_token_budget(msgs, 10000, TokenHandler())
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
    result = enforce_tool_token_budget(msgs, 2500, TokenHandler())

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
    result = enforce_tool_token_budget(msgs, 20, TokenHandler())

    assert result[0].additional_kwargs["truncated"] is True
    assert result[1].additional_kwargs["truncated"] is True
    assert "[OUTPUT TRUNCATED" in result[0].content
    assert "[OUTPUT TRUNCATED" in result[1].content


def test_enforce_tool_token_budget_preserves_metadata():
    """Test that tool_call_id and status are preserved after truncation."""
    msgs = [_make_tool_message("word " * 1000, "my_call_id")]
    msgs[0].status = "success"
    result = enforce_tool_token_budget(msgs, 20, TokenHandler())

    assert result[0].tool_call_id == "my_call_id"
    assert result[0].status == "success"


@pytest.mark.asyncio
async def test_execute_tool_calls_stream_divides_budget_per_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify each tool receives budget/N, not the full budget."""
    budgets_seen: list[int] = []

    original_execute_with_retries = tools_module._execute_with_retries

    async def _spy_execute(
        *, tool, tool_args, tools_token_budget
    ) -> tuple[str, str, bool, dict | None]:
        budgets_seen.append(tools_token_budget)
        return await original_execute_with_retries(
            tool=tool, tool_args=tool_args, tools_token_budget=tools_token_budget
        )

    monkeypatch.setattr(tools_module, "_execute_with_retries", _spy_execute)
    monkeypatch.setattr(tools_module, "need_validation", lambda **kwargs: False)

    total_budget = 9000
    tool_calls = [
        ("call_1", {}, FakeTool("tool1")),
        ("call_2", {}, FakeTool("tool2")),
        ("call_3", {}, FakeTool("tool3")),
    ]

    _ = [
        event
        async for event in execute_tool_calls_stream(
            tool_calls, tools_token_budget=total_budget, streaming=False
        )
    ]

    assert len(budgets_seen) == 3
    expected_per_tool = total_budget // 3
    for budget in budgets_seen:
        assert budget == expected_per_tool, (
            f"Tool received budget {budget}, expected {expected_per_tool} "
            f"(total {total_budget} / 3 tools)"
        )
