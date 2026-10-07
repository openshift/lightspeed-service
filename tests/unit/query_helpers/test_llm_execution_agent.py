"""Unit tests for LLMExecutionAgent."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools.structured import StructuredTool
from opentelemetry.trace import SpanKind
from pydantic import BaseModel

from ols import config, constants

# must be set before importing modules that pull in auth
config.ols_config.authentication_config.module = "k8s"

from ols.app.models.models import StreamChunkType, StreamedChunk  # noqa: E402
from ols.src.prompts import prompts  # noqa: E402
from ols.src.prompts.prompt_generator import GeneratePrompt  # noqa: E402
from ols.src.query_helpers.llm_execution_agent import (  # noqa: E402
    FINAL_SYNTHESIS_FAILURE,
    FINAL_SYNTHESIS_INSTRUCTION,
    LLMExecutionAgent,
    RoundLLMResult,
)
from ols.src.tools.tool_result_inspection import (  # noqa: E402
    ToolResultInspectionError,
    ToolResultRejectedError,
)
from ols.src.tools.tools import (  # noqa: E402
    ApprovalRequiredEvent,
    ToolResultBudgetExceededError,
    ToolResultEvent,
    _approval_rejection_event,
)
from ols.utils.audit_logger import AuditContext, AuditLogger  # noqa: E402
from ols.utils.token_handler import (  # noqa: E402
    TokenBudgetTracker,
    TokenCategory,
    TokenHandler,
)
from tests.mock_classes.mock_llm_loader import MockLLMLoader  # noqa: E402
from tests.mock_classes.mock_tools import mock_tools_map  # noqa: E402
from tests.unit.conftest import make_audit_ctx  # noqa: E402


class SampleTool(StructuredTool):
    """Simple structured tool for deduplication tests."""

    def __init__(self, name: str, description: str = "sample tool") -> None:
        """Initialize simple fake structured tool."""

        class _Schema(BaseModel):
            pass

        async def _coro(**kwargs):  # type: ignore [no-untyped-def]
            return "ok"

        super().__init__(
            name=name,
            description=description,
            func=lambda **kwargs: "ok",
            coroutine=_coro,
            args_schema=_Schema,
        )


@pytest.fixture(scope="function", autouse=True)
def _setup():
    """Set up config for tests."""
    config.reload_from_yaml_file("tests/config/valid_config_without_mcp.yaml")


def _make_agent(**overrides: object) -> LLMExecutionAgent:
    """Create an LLMExecutionAgent with sensible test defaults."""
    model_config = MagicMock()
    model_config.max_tokens_for_tools = 50000
    model_config.context_window_size = 200_000
    model_config.parameters.max_tokens_for_response = 8000
    tracker = TokenBudgetTracker(
        token_handler=TokenHandler(),
        context_window_size=model_config.context_window_size,
        max_response_tokens=model_config.parameters.max_tokens_for_response,
        max_tool_tokens=model_config.max_tokens_for_tools,
        round_cap_fraction=config.ols_config.tool_round_cap_fraction,
    )
    tracker.set_tool_loop_max_rounds(10)
    defaults: dict[str, object] = {
        "bare_llm": MockLLMLoader(),
        "model": "mock_model",
        "provider": "mock_provider",
        "provider_type": "mock_type",
        "model_config": model_config,
        "streaming": False,
        "token_budget_tracker": tracker,
    }
    defaults.update(overrides)
    return LLMExecutionAgent(**defaults)


@pytest.mark.asyncio
async def test_inspect_tool_messages_checks_every_result_before_delivery() -> None:
    """Inspect all tool messages before the loop exposes any result."""
    classifier = MagicMock()
    classifier.inspect = AsyncMock()
    agent = _make_agent(tool_result_classifier=classifier)
    agent.model_config.context_window_size = 613
    agent.model_config.parameters.max_tokens_for_response = 100
    messages = [
        ToolMessage(content="first", tool_call_id="call-1"),
        ToolMessage(content="second", tool_call_id="call-2", status="error"),
    ]

    await agent._inspect_tool_messages(
        messages, {"call-1": "first_tool", "call-2": "second_tool"}
    )

    assert classifier.inspect.await_count == 2
    assert classifier.inspect.await_args_list[0].args[:3] == (
        "first_tool",
        "result",
        "first",
    )
    assert classifier.inspect.await_args_list[1].args[:3] == (
        "second_tool",
        "error",
        "second",
    )
    assert classifier.inspect.await_args_list[0].kwargs["overlap_tokens"] == 0


@pytest.mark.parametrize("outcome", ["timeout", "rejected"])
@pytest.mark.asyncio
async def test_approval_decisions_stream_without_classifying_service_generated_text(
    outcome: str,
) -> None:
    """Emit a safe approval decision even when its wording would fail inspection."""
    classifier = MagicMock()
    classifier.inspect = AsyncMock(
        side_effect=ToolResultRejectedError("tool_manipulation")
    )
    agent = _make_agent(tool_result_classifier=classifier)
    messages: list = []

    async def _fake_execute(*args: object, **kwargs: object) -> AsyncGenerator:
        yield ApprovalRequiredEvent(
            data={
                "approval_id": "approval-1",
                "tool_name": "get_namespaces_mock",
                "tool_description": "desc",
                "tool_args": {},
                "tool_annotation": {},
            }
        )
        yield _approval_rejection_event(tool_call_id="call-1", outcome=outcome)

    tool_call_chunks = [
        AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_calls=[{"name": "get_namespaces_mock", "args": {}, "id": "call-1"}],
        )
    ]
    with patch(
        "ols.src.query_helpers.llm_execution_agent.execute_tool_calls_stream",
        side_effect=_fake_execute,
    ):
        streamed = [
            chunk
            async for chunk in agent._process_tool_calls_for_round(
                round_index=1,
                tool_call_chunks=tool_call_chunks,
                all_chunks=[],
                all_tools_dict={"get_namespaces_mock": mock_tools_map[0]},
                duplicate_tool_names=set(),
                messages=messages,
            )
        ]

    assert [chunk.type for chunk in streamed] == [
        StreamChunkType.TOOL_CALL,
        StreamChunkType.APPROVAL_REQUIRED,
        StreamChunkType.TOOL_RESULT,
    ]
    assert streamed[-1].data["status"] == "error"
    assert "Do not retry this exact tool call" in streamed[-1].data["content"]
    classifier.inspect.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_streaming_approval_rejection_does_not_fail_inspection() -> None:
    """Deliver a non-streaming approval rejection without inspecting service text."""
    classifier = MagicMock()
    classifier.inspect = AsyncMock(
        side_effect=ToolResultRejectedError("tool_manipulation")
    )
    agent = _make_agent(tool_result_classifier=classifier)
    tool = SampleTool("approval_tool")
    messages: list = []
    tool_call_chunks = [
        AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_calls=[{"name": tool.name, "args": {}, "id": "call-1"}],
        )
    ]

    with patch("ols.src.tools.tools.need_validation", return_value=True):
        streamed = [
            chunk
            async for chunk in agent._process_tool_calls_for_round(
                round_index=1,
                tool_call_chunks=tool_call_chunks,
                all_chunks=[],
                all_tools_dict={tool.name: tool},
                duplicate_tool_names=set(),
                messages=messages,
            )
        ]

    assert [chunk.type for chunk in streamed] == [
        StreamChunkType.TOOL_CALL,
        StreamChunkType.TOOL_RESULT,
    ]
    assert streamed[-1].data["status"] == "error"
    assert streamed[-1].data["name"] == tool.name
    classifier.inspect.assert_not_awaited()


@pytest.mark.asyncio
async def test_external_error_matching_approval_text_is_still_inspected() -> None:
    """Do not trust a tool response just because it resembles an approval decision."""
    classifier = MagicMock()
    classifier.inspect = AsyncMock(
        side_effect=ToolResultRejectedError("tool_manipulation")
    )
    agent = _make_agent(tool_result_classifier=classifier)
    message = ToolMessage(
        content="Tool approval timed out. Do not retry this exact tool call.",
        status="error",
        tool_call_id="call-1",
    )

    with pytest.raises(ToolResultRejectedError):
        await agent._inspect_tool_messages([message], {"call-1": "get_namespaces_mock"})
    classifier.inspect.assert_awaited_once()


@pytest.mark.asyncio
async def test_inspect_tool_messages_rejects_insufficient_classifier_budget() -> None:
    """Reject inspection when the model context cannot hold a classifier request."""
    classifier = MagicMock()
    classifier.inspect = AsyncMock()
    agent = _make_agent(tool_result_classifier=classifier)
    agent.model_config.context_window_size = 600
    agent.model_config.parameters.max_tokens_for_response = 1000

    with pytest.raises(ToolResultInspectionError, match="context budget"):
        await agent._inspect_tool_messages(
            [ToolMessage(content="result", tool_call_id="call-1")],
            {"call-1": "get_pods"},
        )

    classifier.inspect.assert_not_called()


@pytest.mark.asyncio
async def test_inspect_tool_messages_propagates_rejection() -> None:
    """Stop the tool loop when inspection rejects a result."""
    classifier = MagicMock()
    classifier.inspect = AsyncMock(side_effect=ToolResultRejectedError("rejected"))
    agent = _make_agent(tool_result_classifier=classifier)

    with pytest.raises(ToolResultRejectedError):
        await agent._inspect_tool_messages(
            [ToolMessage(content="unsafe", tool_call_id="call-1")],
            {"call-1": "get_pods"},
        )


@pytest.mark.parametrize(
    ("inspection_error", "expected_outcome", "expected_category"),
    [
        (
            ToolResultRejectedError("rejected", "instruction_override"),
            "malicious",
            "instruction_override",
        ),
        (ToolResultInspectionError("classifier unavailable"), "classifier_error", None),
    ],
)
@pytest.mark.asyncio
async def test_inspection_failure_metadata_is_exported(
    otel_setup, inspection_error, expected_outcome, expected_category
) -> None:
    """Export inspection failure attributes before the inspection span ends."""
    audit_ctx = make_audit_ctx(otel_setup)
    classifier = MagicMock()
    classifier.inspect = AsyncMock(side_effect=inspection_error)
    agent = _make_agent(tool_result_classifier=classifier, audit_ctx=audit_ctx)

    with audit_ctx.span("chat"):
        audit_span = audit_ctx.start_span("execute_tool get_pods")
        with pytest.raises(type(inspection_error)):
            await agent._inspect_tool_messages(
                [ToolMessage(content="unsafe", tool_call_id="call-1")],
                {"call-1": "get_pods"},
                {"call-1": audit_span},
            )
        audit_span.end()

    exporter, _ = otel_setup
    inspection_span = next(
        span for span in exporter.spans if span.name == "tool_result.inspection"
    )
    assert inspection_span.attributes["inspection.outcome"] == expected_outcome
    assert inspection_span.status.status_code.name == "ERROR"
    if expected_category is not None:
        assert inspection_span.attributes["inspection.category"] == expected_category


@pytest.mark.asyncio
async def test_inspect_tool_messages_audits_passing_sibling_after_rejection(
    otel_setup,
) -> None:
    """Audit each passing result without capturing a rejected sibling."""
    classifier = MagicMock()
    classifier.inspect = AsyncMock(
        side_effect=[ToolResultRejectedError("rejected"), None]
    )
    audit_ctx = make_audit_ctx(otel_setup)
    agent = _make_agent(tool_result_classifier=classifier, audit_ctx=audit_ctx)
    audit_spans = {
        "call-rejected": audit_ctx.start_span("execute_tool unsafe_tool"),
        "call-accepted": audit_ctx.start_span("execute_tool safe_tool"),
    }
    tool_messages = [
        ToolMessage(
            content="unsafe-result",
            status="success",
            tool_call_id="call-rejected",
            additional_kwargs={"duration_ms": 12},
        ),
        ToolMessage(
            content="safe-result",
            status="success",
            tool_call_id="call-accepted",
            additional_kwargs={"duration_ms": 34},
        ),
    ]

    with pytest.raises(ToolResultRejectedError):
        await agent._inspect_tool_messages(
            tool_messages,
            {"call-rejected": "unsafe_tool", "call-accepted": "safe_tool"},
            audit_spans,
        )

    assert classifier.inspect.await_count == 2
    for span in audit_spans.values():
        span.end()
    exporter, _ = otel_setup
    spans = {span.name: span for span in exporter.spans}
    assert not any(
        event.name == "tool.result"
        for event in spans["execute_tool unsafe_tool"].events
    )
    accepted_event = next(
        event
        for event in spans["execute_tool safe_tool"].events
        if event.name == "tool.result"
    )
    assert accepted_event.attributes["output"] == "safe-result"


@pytest.mark.asyncio
async def test_concurrent_round_audits_only_individually_inspected_results(otel_setup):
    """Record passing output even when a sibling result is rejected."""
    audit_ctx = make_audit_ctx(otel_setup)
    classifier = MagicMock()
    classifier.inspect = AsyncMock(
        side_effect=[ToolResultRejectedError("rejected"), None]
    )
    agent = _make_agent(tool_result_classifier=classifier, audit_ctx=audit_ctx)
    messages: list = []

    async def _fake_execute(*args, **kwargs):
        for call_id, tool_name, content in (
            ("call-rejected", "unsafe_tool", "unsafe-result"),
            ("call-accepted", "safe_tool", "safe-result"),
        ):
            span = audit_ctx.start_span(
                f"execute_tool {tool_name}", **{"gen_ai.tool.name": tool_name}
            )
            yield ToolResultEvent(
                data=ToolMessage(
                    content=content,
                    status="success",
                    tool_call_id=call_id,
                    name=tool_name,
                    additional_kwargs={"duration_ms": 20},
                ),
                audit_span=span,
            )

    tool_call_chunks = [
        AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_calls=[
                {"name": "unsafe_tool", "args": {}, "id": "call-rejected"},
                {"name": "safe_tool", "args": {}, "id": "call-accepted"},
            ],
        )
    ]

    with audit_ctx.span("chat mock"):
        with patch(
            "ols.src.query_helpers.llm_execution_agent.execute_tool_calls_stream",
            side_effect=_fake_execute,
        ):
            with pytest.raises(ToolResultRejectedError):
                [
                    chunk
                    async for chunk in agent._process_tool_calls_for_round(
                        round_index=1,
                        tool_call_chunks=tool_call_chunks,
                        all_chunks=[],
                        all_tools_dict={
                            "unsafe_tool": SampleTool("unsafe_tool"),
                            "safe_tool": SampleTool("safe_tool"),
                        },
                        duplicate_tool_names=set(),
                        messages=messages,
                    )
                ]

    assert classifier.inspect.await_count == 2
    assert not any(isinstance(message, ToolMessage) for message in messages)
    exporter, _ = otel_setup
    spans = {span.name: span for span in exporter.spans}
    rejected_span = spans["execute_tool unsafe_tool"]
    accepted_span = spans["execute_tool safe_tool"]
    assert not any(event.name == "tool.result" for event in rejected_span.events)
    accepted_event = next(
        event for event in accepted_span.events if event.name == "tool.result"
    )
    assert accepted_event.attributes["output"] == "safe-result"
    assert "unsafe-result" not in str(rejected_span.events)


def test_resolve_tool_call_definitions_targeted_paths():
    """Test targeted paths in _resolve_tool_call_definitions."""
    agent = _make_agent()
    all_tools_dict = {"get_namespaces_mock": mock_tools_map[0]}
    duplicate_tool_names = {"dup_tool"}
    tool_calls: list[dict[str, object]] = [
        {"name": None, "args": {}, "id": "missing_name"},
        {"name": "dup_tool", "args": {}, "id": "duplicate"},
        {"name": "not_found", "args": {}, "id": "unavailable"},
        {"name": "get_namespaces_mock", "args": "bad", "id": "bad_args"},
        {"name": "get_namespaces_mock", "args": {"ok": True}, "id": "valid"},
    ]

    definitions, skipped = agent._resolve_tool_call_definitions(
        tool_calls, all_tools_dict, duplicate_tool_names
    )

    assert len(definitions) == 1
    assert definitions[0][0] == "valid"
    assert definitions[0][1] == {"ok": True}
    assert definitions[0][2] is mock_tools_map[0]
    assert len(skipped) == 4
    skipped_ids = {msg.tool_call_id for msg in skipped}
    assert skipped_ids == {"missing_name", "duplicate", "unavailable", "bad_args"}


def test_resolve_tool_call_definitions_none_args_normalized_to_empty_dict():
    """Test that None tool args are normalized to {}."""
    agent = _make_agent()
    tool = mock_tools_map[0]
    definitions, skipped = agent._resolve_tool_call_definitions(
        [{"name": tool.name, "args": None, "id": "call_none"}],
        {tool.name: tool},
        set(),
    )

    assert skipped == []
    assert len(definitions) == 1
    assert definitions[0][0] == "call_none"
    assert definitions[0][1] == {}
    assert definitions[0][2] is tool


def test_streamed_chunks_from_list_content_text_and_reasoning():
    """Test _streamed_chunks_from_list_content extracts text and reasoning chunks."""
    agent = _make_agent()
    content: list[object] = [
        {"type": "text", "text": "hello"},
        {"type": "reasoning", "summary": [{"text": "thinking"}]},
        "not-a-dict",
        {"type": "unknown"},
        {"type": "text", "text": ""},
        {"type": "reasoning", "summary": [{"text": ""}, "not-a-dict-part"]},
    ]
    chunks = agent._streamed_chunks_from_list_content(
        content, chunk_counter=10, is_final_round=False
    )
    assert len(chunks) == 2
    assert chunks[0].type == StreamChunkType.TEXT
    assert chunks[0].text == "hello"
    assert chunks[1].type == StreamChunkType.REASONING
    assert chunks[1].text == "thinking"


def test_streamed_chunks_from_list_content_multiple_reasoning_parts():
    """Test _streamed_chunks_from_list_content handles multiple reasoning summary parts."""
    agent = _make_agent()
    content = [
        {"type": "reasoning", "summary": [{"text": "step 1"}, {"text": "step 2"}]},
    ]
    chunks = agent._streamed_chunks_from_list_content(
        content, chunk_counter=0, is_final_round=False
    )
    assert len(chunks) == 2
    assert all(c.type == StreamChunkType.REASONING for c in chunks)
    assert chunks[0].text == "step 1"
    assert chunks[1].text == "step 2"


@pytest.mark.asyncio
async def test_collect_round_llm_chunks_targeted_paths():
    """Test _collect_round_llm_chunks yields text and populates result."""
    agent = _make_agent()

    async def _fake_invoke(*args, **kwargs):
        yield AIMessageChunk(content="hello", response_metadata={})
        yield AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_call_chunks=[
                {"name": "get_namespaces_mock", "args": "{}", "id": "call_1"}
            ],
            tool_calls=[{"name": "get_namespaces_mock", "args": {}, "id": "call_1"}],
        )

    with patch.object(agent, "_invoke_llm", side_effect=_fake_invoke):
        result = RoundLLMResult()
        streamed = [
            chunk
            async for chunk in agent._collect_round_llm_chunks(
                messages=[],
                llm_input_values={},
                all_mcp_tools=mock_tools_map,
                is_final_round=False,
                token_counter=AsyncMock(),
                round_index=1,
                result=result,
            )
        ]

    assert result.should_stop is False
    assert len(streamed) == 1
    assert streamed[0].type == StreamChunkType.TEXT
    assert streamed[0].text == "hello"
    assert len(result.tool_call_chunks) == 1
    assert len(result.all_chunks) == 2


@pytest.mark.asyncio
async def test_collect_round_llm_chunks_timeout_without_any_chunks():
    """Test round timeout path when LLM yields nothing before timeout."""
    agent = _make_agent()

    async def _slow_invoke(*args, **kwargs):
        await asyncio.sleep(0.05)
        if False:
            yield AIMessageChunk(content="", response_metadata={})

    with (
        patch(
            "ols.src.query_helpers.llm_execution_agent.constants.TOOL_CALL_ROUND_TIMEOUT",
            0.001,
        ),
        patch.object(agent, "_invoke_llm", side_effect=_slow_invoke),
    ):
        result = RoundLLMResult()
        streamed = [
            chunk
            async for chunk in agent._collect_round_llm_chunks(
                messages=[],
                llm_input_values={},
                all_mcp_tools=mock_tools_map,
                is_final_round=False,
                token_counter=AsyncMock(),
                round_index=1,
                result=result,
            )
        ]

    assert result.should_stop is True
    assert result.tool_call_chunks == []
    assert len(streamed) == 1
    assert streamed[0].type == StreamChunkType.TEXT
    assert "I could not complete this request in time." in streamed[0].text


@pytest.mark.asyncio
async def test_collect_round_llm_chunks_timeout_after_partial_text():
    """Test timeout still preserves already-streamed text before fallback."""
    agent = _make_agent()

    async def _partial_then_slow(*args, **kwargs):
        yield AIMessageChunk(content="partial", response_metadata={})
        await asyncio.sleep(0.05)
        if False:
            yield AIMessageChunk(content="", response_metadata={})

    with (
        patch(
            "ols.src.query_helpers.llm_execution_agent.constants.TOOL_CALL_ROUND_TIMEOUT",
            0.001,
        ),
        patch.object(agent, "_invoke_llm", side_effect=_partial_then_slow),
    ):
        result = RoundLLMResult()
        streamed = [
            chunk
            async for chunk in agent._collect_round_llm_chunks(
                messages=[],
                llm_input_values={},
                all_mcp_tools=mock_tools_map,
                is_final_round=False,
                token_counter=AsyncMock(),
                round_index=1,
                result=result,
            )
        ]

    assert result.should_stop is True
    assert result.tool_call_chunks == []
    assert [c.type for c in streamed] == [StreamChunkType.TEXT, StreamChunkType.TEXT]
    assert streamed[0].text == "partial"
    assert "I could not complete this request in time." in streamed[1].text


@pytest.mark.asyncio
async def test_collect_round_llm_chunks_stop_short_circuits_before_timeout():
    """Test finish_reason=stop returns immediately without timeout fallback."""
    agent = _make_agent()

    async def _stop_immediately(*args, **kwargs):
        yield AIMessageChunk(content="", response_metadata={"finish_reason": "stop"})

    with (
        patch(
            "ols.src.query_helpers.llm_execution_agent.constants.TOOL_CALL_ROUND_TIMEOUT",
            0.001,
        ),
        patch.object(agent, "_invoke_llm", side_effect=_stop_immediately),
    ):
        result = RoundLLMResult()
        streamed = [
            chunk
            async for chunk in agent._collect_round_llm_chunks(
                messages=[],
                llm_input_values={},
                all_mcp_tools=mock_tools_map,
                is_final_round=False,
                token_counter=AsyncMock(),
                round_index=1,
                result=result,
            )
        ]

    assert result.should_stop is True
    assert result.tool_call_chunks == []
    assert streamed == []


@pytest.mark.asyncio
async def test_collect_round_llm_chunks_handles_string_chunk():
    """Test fake-LLM compatibility path where chunk is plain string."""
    agent = _make_agent()

    async def _string_invoke(*args, **kwargs):
        yield "plain-string-chunk"

    with patch.object(agent, "_invoke_llm", side_effect=_string_invoke):
        result = RoundLLMResult()
        streamed = [
            chunk
            async for chunk in agent._collect_round_llm_chunks(
                messages=[],
                llm_input_values={},
                all_mcp_tools=[],
                is_final_round=False,
                token_counter=AsyncMock(),
                round_index=1,
                result=result,
            )
        ]

    assert result.should_stop is False
    assert result.tool_call_chunks == []
    assert len(streamed) == 1
    assert streamed[0].type == StreamChunkType.TEXT
    assert streamed[0].text == "plain-string-chunk"


@pytest.mark.asyncio
async def test_collect_round_llm_chunks_with_reasoning_list_content():
    """Test _collect_round_llm_chunks processes list content with reasoning blocks."""
    agent = _make_agent()

    async def _reasoning_invoke(*args, **kwargs):
        yield AIMessageChunk(
            content=[
                {"type": "reasoning", "summary": [{"text": "thinking hard"}]},
            ],
            response_metadata={},
        )
        yield AIMessageChunk(
            content=[{"type": "text", "text": "the answer"}],
            response_metadata={},
        )
        yield AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "stop"},
        )

    with patch.object(agent, "_invoke_llm", side_effect=_reasoning_invoke):
        result = RoundLLMResult()
        streamed = [
            chunk
            async for chunk in agent._collect_round_llm_chunks(
                messages=[],
                llm_input_values={},
                all_mcp_tools=[],
                is_final_round=True,
                token_counter=AsyncMock(),
                round_index=1,
                result=result,
            )
        ]

    assert result.should_stop is True
    assert result.tool_call_chunks == []
    assert len(result.all_chunks) == 2
    assert len(streamed) == 2
    assert streamed[0].type == StreamChunkType.REASONING
    assert streamed[0].text == "thinking hard"
    assert streamed[1].type == StreamChunkType.TEXT
    assert streamed[1].text == "the answer"


@pytest.mark.asyncio
async def test_process_tool_calls_for_round_propagates_inspection_failure():
    """Propagate inspection failures to the streaming endpoint."""
    classifier = MagicMock()
    classifier.inspect = AsyncMock(
        side_effect=ToolResultInspectionError("classifier failed")
    )
    agent = _make_agent(tool_result_classifier=classifier)
    messages: list = []

    async def _fake_execute(*args, **kwargs):
        yield ToolResultEvent(
            data=ToolMessage(
                content="unsafe",
                status="success",
                tool_call_id="call_1",
            )
        )

    tool_call_chunks = [
        AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_calls=[{"name": "get_namespaces_mock", "args": {}, "id": "call_1"}],
        )
    ]

    with patch(
        "ols.src.query_helpers.llm_execution_agent.execute_tool_calls_stream",
        side_effect=_fake_execute,
    ):
        with pytest.raises(ToolResultInspectionError, match="classifier failed"):
            [
                chunk
                async for chunk in agent._process_tool_calls_for_round(
                    round_index=1,
                    tool_call_chunks=tool_call_chunks,
                    all_chunks=[],
                    all_tools_dict={"get_namespaces_mock": mock_tools_map[0]},
                    duplicate_tool_names=set(),
                    messages=messages,
                )
            ]


@pytest.mark.asyncio
async def test_process_tool_calls_for_round_streams_approval_and_result():
    """Test _process_tool_calls_for_round streams approval + tool_result."""
    agent = _make_agent()
    messages: list = []

    async def _fake_execute(*args, **kwargs):
        yield ApprovalRequiredEvent(
            data={
                "approval_id": "aid-1",
                "tool_name": "get_namespaces_mock",
                "tool_description": "desc",
                "tool_args": {},
                "tool_annotation": {},
            }
        )
        yield ToolResultEvent(
            data=ToolMessage(
                content="ok",
                status="success",
                tool_call_id="call_1",
                additional_kwargs={"truncated": False},
            )
        )

    tool_call_chunks = [
        AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_calls=[{"name": "get_namespaces_mock", "args": {}, "id": "call_1"}],
        )
    ]

    with patch(
        "ols.src.query_helpers.llm_execution_agent.execute_tool_calls_stream",
        side_effect=_fake_execute,
    ):
        streamed = [
            chunk
            async for chunk in agent._process_tool_calls_for_round(
                round_index=1,
                tool_call_chunks=tool_call_chunks,
                all_chunks=[],
                all_tools_dict={"get_namespaces_mock": mock_tools_map[0]},
                duplicate_tool_names=set(),
                messages=messages,
            )
        ]

    assert [chunk.type for chunk in streamed] == [
        StreamChunkType.TOOL_CALL,
        StreamChunkType.APPROVAL_REQUIRED,
        StreamChunkType.TOOL_RESULT,
    ]
    assert streamed[1].data["approval_id"] == "aid-1"
    assert streamed[2].data["type"] == "tool_result"
    assert agent._tracker.usage(TokenCategory.TOOL_RESULT) > 0
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_process_tool_calls_inspects_raw_then_wraps_external_result():
    """Inspect raw content before adding markers to model-facing results."""
    classifier = MagicMock()
    classifier.inspect = AsyncMock()
    agent = _make_agent(tool_result_classifier=classifier)
    messages: list = []

    async def _fake_execute(*args, **kwargs):
        yield ToolResultEvent(
            data=ToolMessage(
                content="pod-a",
                status="success",
                tool_call_id="call_1",
                name="get_namespaces_mock",
                additional_kwargs={"truncated": False},
            )
        )

    tool_call_chunks = [
        AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_calls=[{"name": "get_namespaces_mock", "args": {}, "id": "call_1"}],
        )
    ]

    with patch(
        "ols.src.query_helpers.llm_execution_agent.execute_tool_calls_stream",
        side_effect=_fake_execute,
    ):
        streamed = [
            chunk
            async for chunk in agent._process_tool_calls_for_round(
                round_index=1,
                tool_call_chunks=tool_call_chunks,
                all_chunks=[],
                all_tools_dict={"get_namespaces_mock": mock_tools_map[0]},
                duplicate_tool_names=set(),
                messages=messages,
            )
        ]

    wrapped = '<tool_data source="get_namespaces_mock">\npod-a\n</tool_data>'
    assert classifier.inspect.await_args.args[:3] == (
        "get_namespaces_mock",
        "result",
        "pod-a",
    )
    assert messages[-1].content == wrapped
    assert streamed[-1].data["content"] == "pod-a"


@pytest.mark.asyncio
async def test_process_tool_calls_audits_raw_output_and_delivers_raw_events(otel_setup):
    """Inspect budgeted content but audit and stream the complete raw result."""
    audit_ctx = make_audit_ctx(otel_setup)
    classifier = MagicMock()
    classifier.inspect = AsyncMock()
    agent = _make_agent(tool_result_classifier=classifier, audit_ctx=audit_ctx)
    agent._tracker.max_tool_tokens = 500
    messages: list = []
    raw_content = "pod-a\n" * 1000
    audit_span = audit_ctx.start_span("execute_tool get_namespaces_mock")

    async def _fake_execute(*args, **kwargs):
        yield ToolResultEvent(
            data=ToolMessage(
                content=raw_content,
                status="success",
                tool_call_id="call_raw_audit",
                name="get_namespaces_mock",
                additional_kwargs={"truncated": False, "duration_ms": 4},
            ),
            audit_span=audit_span,
        )

    tool_call_chunks = [
        AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_calls=[
                {
                    "name": "get_namespaces_mock",
                    "args": {},
                    "id": "call_raw_audit",
                }
            ],
        )
    ]

    with patch(
        "ols.src.query_helpers.llm_execution_agent.execute_tool_calls_stream",
        side_effect=_fake_execute,
    ):
        streamed = [
            chunk
            async for chunk in agent._process_tool_calls_for_round(
                round_index=1,
                tool_call_chunks=tool_call_chunks,
                all_chunks=[],
                all_tools_dict={"get_namespaces_mock": mock_tools_map[0]},
                duplicate_tool_names=set(),
                messages=messages,
            )
        ]

    inspected_content = classifier.inspect.await_args.args[2]
    assert inspected_content != raw_content
    assert "<tool_data" not in inspected_content
    assert streamed[-1].data["content"] == raw_content
    model_message = next(
        message for message in messages if isinstance(message, ToolMessage)
    )
    assert inspected_content in model_message.content
    assert model_message.content.startswith('<tool_data source="get_namespaces_mock">')
    assert model_message.additional_kwargs["token_count"] <= 300

    exporter, _ = otel_setup
    tool_span = next(
        span
        for span in exporter.spans
        if span.name == "execute_tool get_namespaces_mock"
    )
    result_event = next(
        event for event in tool_span.events if event.name == "tool.result"
    )
    assert result_event.attributes["output"] == raw_content


@pytest.mark.asyncio
async def test_process_tool_calls_wraps_external_error_result():
    """Wrap tool-generated errors after inspection passes."""
    classifier = MagicMock()
    classifier.inspect = AsyncMock()
    agent = _make_agent(tool_result_classifier=classifier)
    messages: list = []

    async def _fake_execute(*args, **kwargs):
        yield ToolResultEvent(
            data=ToolMessage(
                content="Tool failed: timeout",
                status="error",
                tool_call_id="call_1",
                name="get_namespaces_mock",
                additional_kwargs={"truncated": False},
            )
        )

    tool_call_chunks = [
        AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_calls=[{"name": "get_namespaces_mock", "args": {}, "id": "call_1"}],
        )
    ]

    with patch(
        "ols.src.query_helpers.llm_execution_agent.execute_tool_calls_stream",
        side_effect=_fake_execute,
    ):
        streamed = [
            chunk
            async for chunk in agent._process_tool_calls_for_round(
                round_index=1,
                tool_call_chunks=tool_call_chunks,
                all_chunks=[],
                all_tools_dict={"get_namespaces_mock": mock_tools_map[0]},
                duplicate_tool_names=set(),
                messages=messages,
            )
        ]

    wrapped = (
        '<tool_data source="get_namespaces_mock">\n'
        "Tool failed: timeout\n</tool_data>"
    )
    assert classifier.inspect.await_args.args[:3] == (
        "get_namespaces_mock",
        "error",
        "Tool failed: timeout",
    )
    assert messages[-1].content == wrapped
    assert streamed[-1].data["content"] == "Tool failed: timeout"


@pytest.mark.asyncio
async def test_process_tool_calls_does_not_wrap_internal_skip_messages():
    """Keep locally generated skips outside the external-data boundary."""
    agent = _make_agent()
    messages: list = []
    tool = mock_tools_map[0]
    tool_call_chunks = [
        AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_calls=[
                {"name": "missing_tool", "args": {}, "id": "skip_missing"},
                {"name": tool.name, "args": {}, "id": "skip_budget"},
            ],
        )
    ]

    with patch(
        "ols.src.query_helpers.llm_execution_agent.MIN_TOOL_EXECUTION_TOKENS",
        100_000,
    ):
        streamed = [
            chunk
            async for chunk in agent._process_tool_calls_for_round(
                round_index=1,
                tool_call_chunks=tool_call_chunks,
                all_chunks=[],
                all_tools_dict={tool.name: tool},
                duplicate_tool_names=set(),
                messages=messages,
            )
        ]

    result_messages = messages[1:]
    result_chunks = [
        chunk for chunk in streamed if chunk.type == StreamChunkType.TOOL_RESULT
    ]
    assert len(result_messages) == len(result_chunks) == 2
    assert all(message.name is None for message in result_messages)
    assert all("<tool_data" not in message.content for message in result_messages)
    assert all("<tool_data" not in chunk.data["content"] for chunk in result_chunks)


@pytest.mark.asyncio
async def test_process_tool_calls_for_round_skipped_only_without_execution():
    """Test skipped-only path emits tool_result without calling executor."""
    agent = _make_agent()
    messages: list = []
    tool = mock_tools_map[0]
    tool_call_chunks = [
        AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_calls=[{"name": "missing_tool", "args": {}, "id": "skip_1"}],
        )
    ]

    with patch(
        "ols.src.query_helpers.llm_execution_agent.execute_tool_calls_stream",
        new=AsyncMock(side_effect=AssertionError("executor should not be called")),
    ):
        streamed = [
            chunk
            async for chunk in agent._process_tool_calls_for_round(
                round_index=1,
                tool_call_chunks=tool_call_chunks,
                all_chunks=[],
                all_tools_dict={tool.name: tool},
                duplicate_tool_names=set(),
                messages=messages,
            )
        ]

    assert [chunk.type for chunk in streamed] == [
        StreamChunkType.TOOL_CALL,
        StreamChunkType.TOOL_RESULT,
    ]
    assert streamed[1].data["type"] == "tool_result"
    assert "tool is unavailable" in streamed[1].data["content"]
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_process_tool_calls_for_round_ignores_unexpected_execution_event(caplog):
    """Test unexpected tool execution events are ignored with warning."""
    agent = _make_agent()
    messages: list = []
    caplog.set_level(logging.WARNING)

    async def _fake_execute(*args, **kwargs):
        class _UnexpectedEvent:
            event = "unexpected"
            data = "payload"

        yield _UnexpectedEvent()
        yield ToolResultEvent(
            data=ToolMessage(
                content="ok",
                status="success",
                tool_call_id="call_warn",
                additional_kwargs={"truncated": False},
            )
        )

    tool_call_chunks = [
        AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_calls=[{"name": "get_namespaces_mock", "args": {}, "id": "call_warn"}],
        )
    ]

    with patch(
        "ols.src.query_helpers.llm_execution_agent.execute_tool_calls_stream",
        side_effect=_fake_execute,
    ):
        streamed = [
            chunk
            async for chunk in agent._process_tool_calls_for_round(
                round_index=1,
                tool_call_chunks=tool_call_chunks,
                all_chunks=[],
                all_tools_dict={"get_namespaces_mock": mock_tools_map[0]},
                duplicate_tool_names=set(),
                messages=messages,
            )
        ]

    assert any(
        "Ignoring unexpected tool execution event" in rec.message
        for rec in caplog.records
    )
    assert streamed[-1].type == StreamChunkType.TOOL_RESULT


def test_tool_result_chunk_counts_wrapper_but_streams_raw_content():
    """Charge model-facing boundary tokens without wrapping the client event."""
    agent = _make_agent()
    raw_content = "pod-a"
    tool_name = "get_pods"
    message = ToolMessage(
        content=raw_content,
        status="success",
        tool_call_id="call-1",
        name=tool_name,
    )
    wrapped_content = f'<tool_data source="{tool_name}">\n{raw_content}\n</tool_data>'
    expected_tokens = agent._tracker.count_tokens(wrapped_content)

    token_count, chunk = agent._tool_result_chunk_for_message(
        tool_call_message=message,
        tool_name=tool_name,
        tool=SampleTool(tool_name),
        round_index=1,
    )

    assert token_count == expected_tokens
    assert chunk.data["content"] == raw_content


def test_tool_result_chunk_for_message_preserves_metadata_and_logs_has_meta(caplog):
    """Test tool result chunk contains metadata enrichment and has_meta logging."""
    agent = _make_agent()
    caplog.set_level(logging.DEBUG)
    tool = mock_tools_map[0]
    tool.metadata = {"mcp_server": "server-a", "_meta": {"app": "ui"}}
    message = ToolMessage(
        content="ok",
        status="success",
        tool_call_id="call_meta",
        additional_kwargs={"truncated": False},
    )

    _, chunk = agent._tool_result_chunk_for_message(
        tool_call_message=message,
        tool_name=tool.name,
        tool=tool,
        round_index=1,
    )

    assert chunk.type == StreamChunkType.TOOL_RESULT
    assert chunk.data["server_name"] == "server-a"
    assert chunk.data["tool_meta"] == {"app": "ui"}
    assert '"has_meta": true' in caplog.text


@pytest.mark.asyncio
async def test_iterate_with_tools_deduplicates_tool_names(caplog):
    """Test duplicate MCP tool names are disabled and logged."""
    agent = _make_agent()
    caplog.set_level(logging.ERROR)
    tools = [SampleTool("dup"), SampleTool("dup")]

    async def _mock_collect(**kwargs):  # type: ignore [no-untyped-def]
        kwargs["result"].should_stop = True
        if False:
            yield

    with patch.object(agent, "_collect_round_llm_chunks", new=_mock_collect):
        chunks = [
            chunk
            async for chunk in agent._iterate_with_tools(
                messages=[],
                max_rounds=1,
                llm_input_values={},
                token_counter=AsyncMock(),
                all_mcp_tools=tools,
            )
        ]

    assert chunks == []
    assert "Duplicate MCP tool names detected and disabled" in caplog.text


@pytest.mark.asyncio
async def test_iterate_with_tools_reports_tool_result_budget_failure():
    """Explain when no tool result or truncation notice can fit the budget."""
    agent = _make_agent()
    tool = mock_tools_map[0]
    tool_call_chunks = [
        AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_calls=[{"name": tool.name, "args": {}, "id": "call_budget"}],
        )
    ]

    async def _mock_collect(**kwargs):  # type: ignore [no-untyped-def]
        kwargs["result"].tool_call_chunks = tool_call_chunks
        if False:
            yield

    async def _failing_process(**kwargs):  # type: ignore [no-untyped-def]
        if False:
            yield
        raise ToolResultBudgetExceededError(
            "cannot represent tool results within token budget"
        )

    with (
        patch.object(agent, "_collect_round_llm_chunks", new=_mock_collect),
        patch.object(agent, "_process_tool_calls_for_round", new=_failing_process),
    ):
        chunks = [
            chunk
            async for chunk in agent._iterate_with_tools(
                messages=[],
                max_rounds=2,
                llm_input_values={},
                token_counter=AsyncMock(),
                all_mcp_tools=[tool],
            )
        ]

    assert len(chunks) == 1
    assert chunks[0].type == StreamChunkType.TEXT
    assert "tool results exceeded the remaining token budget" in chunks[0].text.lower()


@pytest.mark.asyncio
async def test_iterate_with_tools_handles_tool_execution_error():
    """Test _iterate_with_tools emits fallback when tool execution raises."""
    agent = _make_agent()
    tool = mock_tools_map[0]
    tool_call_chunks = [
        AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_calls=[{"name": tool.name, "args": {}, "id": "call_error"}],
        )
    ]

    async def _mock_collect(**kwargs):  # type: ignore [no-untyped-def]
        kwargs["result"].tool_call_chunks = tool_call_chunks
        if False:
            yield

    async def _failing_process(**kwargs):  # type: ignore [no-untyped-def]
        if False:
            yield
        raise RuntimeError("MCP server unreachable")

    with (
        patch.object(agent, "_collect_round_llm_chunks", new=_mock_collect),
        patch.object(agent, "_process_tool_calls_for_round", new=_failing_process),
    ):
        chunks = [
            chunk
            async for chunk in agent._iterate_with_tools(
                messages=[],
                max_rounds=2,
                llm_input_values={},
                token_counter=AsyncMock(),
                all_mcp_tools=[tool],
            )
        ]

    assert len(chunks) == 1
    assert chunks[0].type == StreamChunkType.TEXT
    assert "I could not complete this request." in chunks[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", [constants.QueryMode.ASK, constants.QueryMode.TROUBLESHOOTING]
)
@pytest.mark.parametrize("answer", ["Evidence-based answer.", ""])
async def test_iterate_with_tools_final_synthesis_preserves_system_prompt(
    mode: constants.QueryMode,
    answer: str,
) -> None:
    """Keep full system prompt while requesting final answer without bound tools."""
    agent = _make_agent(provider_type=constants.PROVIDER_GOOGLE_VERTEX)
    tool = mock_tools_map[0]
    system_instruction = (
        prompts.TROUBLESHOOTING_SYSTEM_INSTRUCTION
        if mode == constants.QueryMode.TROUBLESHOOTING
        else prompts.QUERY_SYSTEM_INSTRUCTION
    )
    messages, inputs = GeneratePrompt(
        "question", [], [], system_instruction, True, mode
    ).generate_prompt(agent.model)
    original_system = messages.messages[0].prompt.template
    agent._tracker.charge(TokenCategory.TOOL_RESULT, agent._tracker.prompt_budget)
    calls: list[dict[str, Any]] = []

    async def collect(**kwargs: Any) -> AsyncGenerator[StreamedChunk, None]:
        """Request tools until the final round, then emit the configured answer."""
        calls.append(kwargs)
        if len(calls) < 5 or not answer:
            kwargs["result"].tool_call_chunks.append(
                AIMessageChunk(
                    content="",
                    tool_calls=[{"name": tool.name, "args": {}, "id": "call"}],
                )
            )
        if len(calls) == 5 and answer:
            yield StreamedChunk(type=StreamChunkType.TEXT, text=answer)

    async def process(**kwargs: Any) -> AsyncGenerator[StreamedChunk, None]:
        """Append and stream tool evidence for the next round."""
        kwargs["messages"].append(ToolMessage(content="evidence", tool_call_id="call"))
        yield StreamedChunk(
            type=StreamChunkType.TOOL_RESULT, data={"content": "evidence"}
        )

    with (
        patch.object(agent, "_collect_round_llm_chunks", new=collect),
        patch.object(agent, "_process_tool_calls_for_round", new=process),
    ):
        chunks = [
            chunk
            async for chunk in agent._iterate_with_tools(
                messages=messages,
                max_rounds=5,
                llm_input_values=inputs,
                token_counter=AsyncMock(),
                all_mcp_tools=[tool],
            )
        ]

    assert len(calls) == 5
    assert calls[-1]["all_mcp_tools"] == []
    final_messages = calls[-1]["messages"].messages
    assert final_messages[0].prompt.template == original_system
    if mode == constants.QueryMode.TROUBLESHOOTING:
        assert "# TOOL USAGE" in final_messages[0].prompt.template
    else:
        assert (
            "Given the user's query you must decide"
            in final_messages[0].prompt.template
        )
    assert len([msg for msg in final_messages if isinstance(msg, ToolMessage)]) == 4
    assert final_messages[-1].content == FINAL_SYNTHESIS_INSTRUCTION
    assert sum(chunk.type == StreamChunkType.TOOL_RESULT for chunk in chunks) == 4
    assert [chunk.text for chunk in chunks if chunk.type == StreamChunkType.TEXT] == [
        answer or FINAL_SYNTHESIS_FAILURE
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_text", ["", " \n"])
async def test_execute_final_synthesis_invocation_error_emits_fallback_and_end(
    initial_text: str,
) -> None:
    """Emit fallback and END when final synthesis fails before a usable answer."""
    agent = _make_agent(provider_type=constants.PROVIDER_GOOGLE_VERTEX)
    messages, inputs = GeneratePrompt("question").generate_prompt(agent.model)

    async def failing_invoke(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[AIMessageChunk, None]:
        """Emit optional whitespace before simulating an upstream failure."""
        yield AIMessageChunk(content=initial_text)
        raise RuntimeError("upstream unavailable")

    with patch.object(agent, "_invoke_llm", new=failing_invoke):
        chunks = [
            chunk
            async for chunk in agent.execute(
                messages, inputs, 1, [mock_tools_map[0]], [], False
            )
        ]

    text = "".join(chunk.text for chunk in chunks if chunk.type == StreamChunkType.TEXT)
    assert text.strip() == FINAL_SYNTHESIS_FAILURE
    assert chunks[-1].type == StreamChunkType.END
    assert sum(chunk.type == StreamChunkType.END for chunk in chunks) == 1
    assert all(
        chunk.type in (StreamChunkType.TEXT, StreamChunkType.END) for chunk in chunks
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_type,max_rounds,has_tools,initial_text",
    [
        (constants.PROVIDER_GOOGLE_VERTEX, 1, True, "Partial answer."),
        (constants.PROVIDER_GOOGLE_VERTEX, 2, True, ""),
        (constants.PROVIDER_OPENAI, 1, True, ""),
        (constants.PROVIDER_GOOGLE_VERTEX, 1, False, ""),
    ],
)
async def test_execute_propagates_invocation_error_outside_empty_final_synthesis(
    provider_type: str, max_rounds: int, has_tools: bool, initial_text: str
) -> None:
    """Preserve errors after partial answers and outside tool-loop synthesis."""
    agent = _make_agent(provider_type=provider_type)
    messages, inputs = GeneratePrompt("question").generate_prompt(agent.model)
    tools = [mock_tools_map[0]] if has_tools else []

    async def failing_invoke(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[AIMessageChunk, None]:
        """Emit an optional partial answer before an upstream failure."""
        yield AIMessageChunk(content=initial_text)
        raise RuntimeError("upstream unavailable")

    with patch.object(agent, "_invoke_llm", new=failing_invoke):
        stream = agent.execute(messages, inputs, max_rounds, tools, [], False)
        if initial_text:
            chunk = await anext(stream)
            assert chunk.type == StreamChunkType.TEXT
            assert chunk.text == initial_text
        with pytest.raises(RuntimeError, match="upstream unavailable"):
            await anext(stream)


def test_prepare_round_request_returns_none_when_budget_insufficient() -> None:
    """Reject final synthesis when added instruction exceeds remaining budget."""
    agent = _make_agent(provider_type=constants.PROVIDER_GOOGLE_VERTEX)
    messages, _ = GeneratePrompt(
        "question", [], [], prompts.QUERY_SYSTEM_INSTRUCTION, True
    ).generate_prompt(agent.model)
    instruction_tokens = agent._tracker.count_tokens(FINAL_SYNTHESIS_INSTRUCTION)
    agent._tracker.charge(
        TokenCategory.PROMPT, agent._tracker.remaining - instruction_tokens + 1
    )

    assert agent._prepare_round_request(messages, [mock_tools_map[0]], True) is None


@pytest.mark.asyncio
async def test_iterate_with_tools_breaks_when_no_tool_calls():
    """Test _iterate_with_tools exits when model emits no tool calls."""
    agent = _make_agent()
    tool = mock_tools_map[0]
    call_count = 0

    async def _mock_collect(**kwargs):  # type: ignore [no-untyped-def]
        nonlocal call_count
        call_count += 1
        yield StreamedChunk(type=StreamChunkType.TEXT, text="answer")

    with patch.object(agent, "_collect_round_llm_chunks", new=_mock_collect):
        chunks = [
            chunk
            async for chunk in agent._iterate_with_tools(
                messages=[],
                max_rounds=10,
                llm_input_values={},
                token_counter=AsyncMock(),
                all_mcp_tools=[tool],
            )
        ]

    assert call_count == 1, (
        "Loop must exit after first round when model emits no tool calls, "
        f"but ran {call_count} rounds"
    )
    assert len(chunks) == 1
    assert chunks[0].text == "answer"


def test_skip_special_chunk_granite_tool_call_sequence():
    """Test skip_special_chunk filters granite tool-call preamble tokens."""
    from ols.src.query_helpers.llm_execution_agent import skip_special_chunk

    granite_model = "granite-3.1-8b"
    expected = ["", "<", "tool", "_", "call", ">"]
    for counter, text in enumerate(expected):
        assert skip_special_chunk(
            text, counter, granite_model, final_round=False
        ), f"Expected chunk {counter} ('{text}') to be skipped for granite"

    assert not skip_special_chunk("hello", 0, granite_model, final_round=False)
    assert not skip_special_chunk("<", 0, granite_model, final_round=False)
    assert not skip_special_chunk("", 0, granite_model, final_round=True)


def test_skip_special_chunk_non_granite_never_skips():
    """Test skip_special_chunk always returns False for non-granite models."""
    from ols.src.query_helpers.llm_execution_agent import skip_special_chunk

    assert not skip_special_chunk("", 0, "gpt-4o", final_round=False)
    assert not skip_special_chunk("<", 1, "gpt-4o", final_round=False)


@pytest.mark.asyncio
async def test_execute_emits_end_chunk_with_rag_and_truncated():
    """Test execute yields an END chunk containing rag_chunks and truncated."""
    agent = _make_agent()
    rag_chunks = [MagicMock(spec=["text", "doc_url"])]

    async def _mock_iterate(**kwargs):  # type: ignore [no-untyped-def]
        yield StreamedChunk(type=StreamChunkType.TEXT, text="response")

    with patch.object(agent, "_iterate_with_tools", new=_mock_iterate):
        chunks = [
            chunk
            async for chunk in agent.execute(
                messages=[],
                llm_input_values={},
                max_rounds=1,
                all_mcp_tools=[],
                rag_chunks=rag_chunks,
                truncated=True,
            )
        ]

    assert len(chunks) == 2
    assert chunks[0].type == StreamChunkType.TEXT
    assert chunks[0].text == "response"
    assert chunks[1].type == StreamChunkType.END
    assert chunks[1].data["rag_chunks"] is rag_chunks
    assert chunks[1].data["truncated"] is True
    assert "token_counter" in chunks[1].data


class TestGenAISpanNaming:
    """Verify LLM turn spans use GenAI semantic convention names and attributes."""

    def test_emit_turn_audit_emits_genai_events(self, otel_setup) -> None:
        """Verify _emit_turn_audit emits gen_ai.choice events and token attributes."""
        audit_ctx = make_audit_ctx(otel_setup)
        agent = _make_agent(audit_ctx=audit_ctx)

        token_counter = MagicMock()
        token_counter.token_counter.input_tokens = 100
        token_counter.token_counter.output_tokens = 40
        token_counter.token_counter.reasoning_tokens = 10

        result = RoundLLMResult()
        result.collected_thinking = ["thinking hard"]
        result.collected_text = ["the answer"]

        with audit_ctx.span("chat mock_model", kind=SpanKind.CLIENT):
            agent._emit_turn_audit(result, 1, token_counter, 0, 0)

        exporter, _ = otel_setup
        span = exporter.spans[0]
        events = {e.name: e for e in span.events}
        assert "gen_ai.choice" in events
        choice_events = [e for e in span.events if e.name == "gen_ai.choice"]
        reasoning_events = [
            e for e in choice_events if "gen_ai.reasoning_content" in e.attributes
        ]
        completion_events = [
            e for e in choice_events if "gen_ai.completion" in e.attributes
        ]
        assert len(reasoning_events) == 1
        assert (
            reasoning_events[0].attributes["gen_ai.reasoning_content"]
            == "thinking hard"
        )
        assert len(completion_events) == 1
        assert completion_events[0].attributes["gen_ai.completion"] == "the answer"
        assert span.attributes["gen_ai.usage.input_tokens"] == 100
        assert span.attributes["gen_ai.usage.output_tokens"] == 50

    def test_emit_turn_audit_skips_empty_thinking(self, otel_setup) -> None:
        """Verify _emit_turn_audit skips thinking event when no reasoning content."""
        audit_ctx = make_audit_ctx(otel_setup)
        agent = _make_agent(audit_ctx=audit_ctx)

        token_counter = MagicMock()
        token_counter.token_counter.input_tokens = 50
        token_counter.token_counter.output_tokens = 20
        token_counter.token_counter.reasoning_tokens = 0

        result = RoundLLMResult()
        result.collected_thinking = []
        result.collected_text = ["just text"]

        with audit_ctx.span("chat mock_model", kind=SpanKind.CLIENT):
            agent._emit_turn_audit(result, 1, token_counter, 0, 0)

        exporter, _ = otel_setup
        span = exporter.spans[0]
        choice_events = [e for e in span.events if e.name == "gen_ai.choice"]
        assert len(choice_events) == 1
        assert "gen_ai.completion" in choice_events[0].attributes
        assert all(
            "gen_ai.reasoning_content" not in e.attributes for e in choice_events
        )

    def test_chat_span_has_genai_attributes(self, otel_setup) -> None:
        """Verify the chat span carries gen_ai.* attributes and correct kind."""
        audit_ctx = make_audit_ctx(otel_setup)
        with audit_ctx.span(
            "chat gpt-4",
            kind=SpanKind.CLIENT,
            **{
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "gpt-4",
                "gen_ai.provider.name": "openai",
            },
            turn_index=1,
        ):
            pass

        exporter, _ = otel_setup
        span = exporter.spans[0]
        assert span.name == "chat gpt-4"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes["gen_ai.operation.name"] == "chat"
        assert span.attributes["gen_ai.request.model"] == "gpt-4"
        assert span.attributes["gen_ai.provider.name"] == "openai"
        assert span.attributes["gen_ai.conversation.id"] == "conv-test"
        assert span.attributes["user_id"] == "user-test"
        assert span.attributes["turn_index"] == 1

    def test_emit_turn_audit_respects_capture_content_false(self, otel_setup) -> None:
        """Verify gen_ai.choice events omit content when capture_content is False."""
        _, tracer = otel_setup
        audit_ctx = AuditContext(
            conversation_id="conv-test",
            user_id="user-test",
            logger=AuditLogger(enabled=True),
            tracer=tracer,
            capture_content=False,
        )
        agent = _make_agent(audit_ctx=audit_ctx)

        token_counter = MagicMock()
        token_counter.token_counter.input_tokens = 10
        token_counter.token_counter.output_tokens = 5
        token_counter.token_counter.reasoning_tokens = 0

        result = RoundLLMResult()
        result.collected_thinking = ["secret thoughts"]
        result.collected_text = ["secret answer"]

        with audit_ctx.span("chat mock_model", kind=SpanKind.CLIENT):
            agent._emit_turn_audit(result, 1, token_counter, 0, 0)

        exporter, _ = otel_setup
        span = exporter.spans[0]
        choice_events = [e for e in span.events if e.name == "gen_ai.choice"]
        assert len(choice_events) == 2
        for e in choice_events:
            assert "gen_ai.completion" not in e.attributes
            assert "gen_ai.reasoning_content" not in e.attributes


@pytest.mark.asyncio
async def test_invoke_llm_observes_duration_histogram():
    """LLM invocation records gen_ai_client_operation_duration_seconds histogram."""
    from langchain_core.prompts import ChatPromptTemplate

    from ols.app.metrics.metrics import gen_ai_client_operation_duration_seconds
    from ols.app.metrics.token_counter import GenericTokenCounter

    agent = _make_agent()
    token_counter = GenericTokenCounter(agent.bare_llm)

    labeled = gen_ai_client_operation_duration_seconds.labels(
        gen_ai_request_model="mock_model",
        gen_ai_provider_name="mock_type",
        gen_ai_operation_name="chat",
    )
    before = [s for s in labeled._samples() if s.name.endswith("_count")]
    count_before = before[0].value if before else 0.0

    messages = ChatPromptTemplate.from_messages(
        [
            ("system", "test"),
            ("human", "{query}"),
        ]
    )
    async for _ in agent._invoke_llm(
        messages=messages,
        llm_input_values={"query": "hello"},
        tools_map=[],
        is_final_round=False,
        token_counter=token_counter,
    ):
        pass

    after = [s for s in labeled._samples() if s.name.endswith("_count")]
    count_after = after[0].value if after else 0.0
    assert count_after - count_before == 1
