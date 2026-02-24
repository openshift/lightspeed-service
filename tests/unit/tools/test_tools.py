"""Unit tests for tools module."""

import asyncio
import time
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel

from ols.src.tools import tools as tools_module
from ols.src.tools.tools import (
    SENSITIVE_KEYWORDS,
    ToolResultEvent,
    _extract_text_from_tool_output,
    _is_retryable_tool_error,
    _jsonify,
    execute_tool_call,
    execute_tool_calls_stream,
    raise_for_sensitive_tool_args,
)


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


async def _collect_tool_messages(
    tool_calls: list[tuple[str, dict[str, Any], StructuredTool]],
    max_tokens_per_output: int = 4000,
) -> list[ToolMessage]:
    """Execute tool stream and collect only tool_result messages."""
    return [
        event.data
        async for event in execute_tool_calls_stream(
            tool_calls, max_tokens_per_output=max_tokens_per_output, streaming=False
        )
        if event.event == "tool_result"
    ]


def test_execute_tool_call_success() -> None:
    """Test execute_tool_call success path."""

    async def _run() -> None:
        status, output, was_truncated = await execute_tool_call(
            FakeTool("fake_tool"), {"a": 1}
        )
        assert output == "fake_output_from_fake_tool"
        assert status == "success"
        assert was_truncated is False

    asyncio.run(_run())


def test_execute_tool_call_failure_raises() -> None:
    """Test execute_tool_call raises tool exception."""

    async def _run() -> None:
        try:
            await execute_tool_call(FakeTool("bad", should_fail=True), {})
            raise AssertionError("Expected exception was not raised")
        except Exception as err:
            assert "Tool execution failed" in str(err)

    asyncio.run(_run())


def test_execute_tool_calls_stream_empty() -> None:
    """Test execute_tool_calls_stream with empty list."""

    async def _run() -> None:
        events = [
            event async for event in execute_tool_calls_stream([], streaming=False)
        ]
        assert events == []

    asyncio.run(_run())


def test_execute_tool_calls_stream_parallel_execution() -> None:
    """Test that tool streams execute in parallel."""

    async def _run() -> None:
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

    asyncio.run(_run())


def test_raise_for_sensitive_tool_args() -> None:
    """Test sensitive argument guard."""
    raise_for_sensitive_tool_args({"tool_args": "normal_args"})
    for keyword in SENSITIVE_KEYWORDS:
        try:
            raise_for_sensitive_tool_args({"tool_args": keyword})
            raise AssertionError("Expected sensitive-args exception")
        except ValueError as err:
            assert "Sensitive keyword in tool arguments" in str(err)


def test_execute_sensitive_tool_calls() -> None:
    """Test sensitive args produce non-retryable synthetic result."""

    async def _run() -> None:
        tool_calls = [
            (
                "tool_call_1",
                {"tool_arg": SENSITIVE_KEYWORDS[0]},
                FakeTool("some_tool"),
            )
        ]
        tool_messages = await _collect_tool_messages(tool_calls)
        assert len(tool_messages) == 1
        assert tool_messages[0].status == "error"
        assert "Sensitive keyword" in tool_messages[0].content
        assert "Do not retry this exact tool call" in tool_messages[0].content

    asyncio.run(_run())


def test_execute_tool_calls_mixed_success_and_failure() -> None:
    """Test mixed successful and failing tool calls."""

    async def _run() -> None:
        tool_calls = [
            ("call_1", {}, FakeTool("success_tool")),
            ("call_2", {}, FakeTool("fail_tool", should_fail=True)),
        ]
        tool_messages = await _collect_tool_messages(tool_calls)
        assert len(tool_messages) == 2
        by_id = {m.tool_call_id: m for m in tool_messages}
        assert by_id["call_1"].status == "success"
        assert by_id["call_1"].content == "fake_output_from_success_tool"
        # tools module returns non-retryable synthetic error on terminal failure
        assert by_id["call_2"].status == "error"
        assert "Do not retry this exact tool call" in by_id["call_2"].content

    asyncio.run(_run())


def test_execute_tool_call_with_truncation() -> None:
    """Test large tool outputs are truncated."""

    async def _run() -> None:
        status, output, was_truncated = await execute_tool_call(
            LargeOutputTool(name="large_tool", output_size=5000),
            {},
            max_tokens=100,
        )
        assert status == "success"
        assert was_truncated is True
        assert "[OUTPUT TRUNCATED" in output

    asyncio.run(_run())


def test_execute_tool_call_no_truncation_small_output() -> None:
    """Test small outputs are not truncated."""

    async def _run() -> None:
        status, output, was_truncated = await execute_tool_call(
            LargeOutputTool(name="small_tool", output_size=10),
            {},
            max_tokens=1000,
        )
        assert status == "success"
        assert was_truncated is False
        assert "[OUTPUT TRUNCATED" not in output

    asyncio.run(_run())


def test_execute_tool_calls_truncation_in_additional_kwargs() -> None:
    """Test truncation flag is stored in additional_kwargs."""

    async def _run() -> None:
        tool_calls = [
            ("call_large", {}, LargeOutputTool(name="large_tool", output_size=5000)),
            ("call_small", {}, LargeOutputTool(name="small_tool", output_size=10)),
        ]
        tool_messages = await _collect_tool_messages(
            tool_calls, max_tokens_per_output=100
        )
        by_id = {m.tool_call_id: m for m in tool_messages}
        assert by_id["call_large"].additional_kwargs.get("truncated") is True
        assert "[OUTPUT TRUNCATED" in by_id["call_large"].content
        assert by_id["call_small"].additional_kwargs.get("truncated") is False
        assert "[OUTPUT TRUNCATED" not in by_id["call_small"].content

    asyncio.run(_run())


def test_extract_text_from_tool_output_string() -> None:
    """Test _extract_text_from_tool_output with plain string."""
    assert _extract_text_from_tool_output("hello") == "hello"


def test_extract_text_from_tool_output_content_blocks() -> None:
    """Test _extract_text_from_tool_output with content blocks."""
    blocks = [
        {"type": "text", "text": "pod1 Running", "id": "lc_1"},
        {"type": "text", "text": "pod2 Running", "id": "lc_2"},
    ]
    assert _extract_text_from_tool_output(blocks) == "pod1 Running\npod2 Running"


def test_extract_text_from_tool_output_single_content_block() -> None:
    """Test _extract_text_from_tool_output with one content block."""
    blocks = [{"type": "text", "text": "some output", "id": "lc_1"}]
    assert _extract_text_from_tool_output(blocks) == "some output"


def test_extract_text_from_tool_output_mixed_list() -> None:
    """Test _extract_text_from_tool_output with mixed list elements."""
    blocks = [{"type": "text", "text": "text content"}, "plain string"]
    assert _extract_text_from_tool_output(blocks) == "text content\nplain string"


def test_extract_text_from_tool_output_non_string_non_list() -> None:
    """Test _extract_text_from_tool_output with unexpected types."""
    assert _extract_text_from_tool_output(42) == "42"
    assert _extract_text_from_tool_output(None) == "None"


def test_extract_text_from_tool_output_empty_list() -> None:
    """Test _extract_text_from_tool_output with empty list."""
    assert _extract_text_from_tool_output([]) == ""


def test_extract_text_from_tool_output_list_with_unknown_item() -> None:
    """Test _extract_text_from_tool_output stringifies unknown list entries."""
    blocks = [{"type": "text", "text": "text content"}, 123]
    assert _extract_text_from_tool_output(blocks) == "text content\n123"


def test_execute_tool_call_with_content_blocks() -> None:
    """Test execute_tool_call handles content block output."""

    async def _run() -> None:
        status, output, was_truncated = await execute_tool_call(
            ContentBlockTool(name="content_tool"),
            {},
        )
        assert status == "success"
        assert output == "pod1 Running\npod2 Running"
        assert was_truncated is False

    asyncio.run(_run())


def test_execute_tool_call_with_content_and_artifact_tuple() -> None:
    """Test execute_tool_call handles content+artifact tuple output."""

    async def _run() -> None:
        status, output, was_truncated = await execute_tool_call(
            TupleOutputTool(name="tuple_tool"),
            {},
        )
        assert status == "success"
        assert output == "tuple content"
        assert was_truncated is False

    asyncio.run(_run())


def test_is_retryable_tool_error_branches() -> None:
    """Test retryability classifier for exception-type branches."""
    assert _is_retryable_tool_error(TimeoutError("timeout")) is True
    assert _is_retryable_tool_error(OSError("os issue")) is True
    assert _is_retryable_tool_error(Exception("hard failure")) is False


def test_jsonify_invalid_json_keeps_original_string() -> None:
    """Test _jsonify keeps original value when json.loads fails."""
    raw = {"payload": "{not-json}", "plain": "value"}
    converted = _jsonify(raw)
    assert converted["payload"] == "{not-json}"
    assert converted["plain"] == "value"


def test_jsonify_non_json_string_kept_as_is() -> None:
    """Test _jsonify keeps non-JSON-looking strings unchanged."""
    raw = {"payload": "not-json"}
    converted = _jsonify(raw)
    assert converted == raw


def test_execute_tool_calls_stream_emits_approval_required_then_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test approval flow emits approval_required event before tool_result."""

    async def _approved(*args: object, **kwargs: object) -> str:
        return "approved"

    monkeypatch.setattr(tools_module, "need_validation", lambda **kwargs: True)
    monkeypatch.setattr(tools_module, "get_approval_decision", _approved)

    async def _run() -> None:
        tool = FakeTool(
            "approved_tool",
            metadata={"annotations": {"readOnlyHint": True, "category": "safe"}},
        )
        events = [
            event
            async for event in execute_tool_calls_stream(
                [("call_approved", {}, tool)],
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

    asyncio.run(_run())


def test_execute_tool_calls_stream_rejection_returns_terminal_result(
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

    async def _run() -> None:
        events = [
            event
            async for event in execute_tool_calls_stream(
                [("call_rejected", {}, FakeTool("rejected_tool"))],
                streaming=True,
            )
        ]

        assert len(events) == 2
        assert events[0].event == "approval_required"
        assert events[1].event == "tool_result"
        assert events[1].data.status == "error"
        assert "execution was rejected" in events[1].data.content
        assert "Do not retry this exact tool call." in events[1].data.content

    asyncio.run(_run())


def test_execute_tool_calls_stream_timeout_returns_timeout_result(
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

    async def _run() -> None:
        events = [
            event
            async for event in execute_tool_calls_stream(
                [("call_timeout", {}, FakeTool("timeout_tool"))],
                streaming=True,
            )
        ]

        assert len(events) == 2
        assert events[0].event == "approval_required"
        assert events[1].event == "tool_result"
        assert events[1].data.status == "error"
        assert "approval timed out" in events[1].data.content

    asyncio.run(_run())


def test_execute_tool_calls_stream_no_approval_when_not_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test no approval event is emitted when validation is not required."""

    async def _must_not_wait(*args: object, **kwargs: object) -> str:
        raise AssertionError(
            "get_approval_decision should not run when approval is not needed"
        )

    monkeypatch.setattr(tools_module, "need_validation", lambda **kwargs: False)
    monkeypatch.setattr(tools_module, "get_approval_decision", _must_not_wait)

    async def _run() -> None:
        events = [
            event
            async for event in execute_tool_calls_stream(
                [("call_no_approval", {}, FakeTool("plain_tool"))],
                streaming=False,
            )
        ]

        assert [event.event for event in events] == ["tool_result"]
        assert events[0].data.content == "fake_output_from_plain_tool"

    asyncio.run(_run())


def test_execute_tool_calls_stream_retries_retryable_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test retry loop retries transient error and eventually succeeds."""

    async def _execute_with_one_retry(
        tool: StructuredTool,
        tool_args: dict[str, Any],
        max_tokens: int,
    ) -> tuple[str, str, bool]:
        count = getattr(_execute_with_one_retry, "count", 0) + 1
        _execute_with_one_retry.count = count
        if count == 1:
            raise TimeoutError("temporary timeout")
        return "success", "retried-ok", False

    async def _no_sleep(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(tools_module, "need_validation", lambda **kwargs: False)
    monkeypatch.setattr(tools_module, "execute_tool_call", _execute_with_one_retry)
    monkeypatch.setattr(tools_module.asyncio, "sleep", _no_sleep)

    async def _run() -> None:
        events = [
            event
            async for event in execute_tool_calls_stream(
                [("call_retry", {}, FakeTool("retry_tool"))],
                streaming=False,
            )
        ]
        assert len(events) == 1
        assert events[0].event == "tool_result"
        assert events[0].data.content == "retried-ok"

    asyncio.run(_run())


def test_execute_tool_calls_stream_cancels_remaining_tasks_on_early_break(
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

    async def _run() -> None:
        gen = execute_tool_calls_stream(
            [
                ("call_fast", {}, FakeTool("tool1")),
                ("call_slow", {}, FakeTool("tool2")),
            ],
            streaming=False,
        )
        first_event = await gen.__anext__()
        assert first_event.event == "tool_result"
        assert first_event.data.tool_call_id == "call_fast"
        await gen.aclose()

    asyncio.run(_run())
