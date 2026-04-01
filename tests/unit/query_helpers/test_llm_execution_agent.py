"""Unit tests for LLMExecutionAgent."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel

from ols import config

# must be set before importing modules that pull in auth
config.ols_config.authentication_config.module = "k8s"

from ols.app.models.models import StreamChunkType, StreamedChunk  # noqa: E402
from ols.src.query_helpers.llm_execution_agent import (  # noqa: E402
    LLMExecutionAgent,
    RoundLLMResult,
    ToolTokenUsage,
)
from ols.src.tools.tools import ApprovalRequiredEvent, ToolResultEvent  # noqa: E402
from ols.utils.token_handler import TokenHandler  # noqa: E402
from tests.mock_classes.mock_llm_loader import MockLLMLoader  # noqa: E402
from tests.mock_classes.mock_tools import mock_tools_map  # noqa: E402


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
    defaults: dict[str, object] = {
        "bare_llm": MockLLMLoader(),
        "model": "mock_model",
        "provider": "mock_provider",
        "provider_type": "mock_type",
        "model_config": model_config,
        "streaming": False,
    }
    defaults.update(overrides)
    return LLMExecutionAgent(**defaults)


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
async def test_process_tool_calls_for_round_streams_approval_and_result():
    """Test _process_tool_calls_for_round streams approval + tool_result."""
    agent = _make_agent()
    token_usage = ToolTokenUsage(used=0)
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
                token_handler=TokenHandler(),
                tool_token_usage=token_usage,
                max_tokens_for_tools=1000,
            )
        ]

    assert [chunk.type for chunk in streamed] == [
        StreamChunkType.TOOL_CALL,
        StreamChunkType.APPROVAL_REQUIRED,
        StreamChunkType.TOOL_RESULT,
    ]
    assert streamed[1].data["approval_id"] == "aid-1"
    assert streamed[2].data["type"] == "tool_result"
    assert token_usage.used > 0
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_process_tool_calls_for_round_skipped_only_without_execution():
    """Test skipped-only path emits tool_result without calling executor."""
    agent = _make_agent()
    token_usage = ToolTokenUsage(used=0)
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
                token_handler=TokenHandler(),
                tool_token_usage=token_usage,
                max_tokens_for_tools=1000,
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
    token_usage = ToolTokenUsage(used=0)
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
                token_handler=TokenHandler(),
                tool_token_usage=token_usage,
                max_tokens_for_tools=1000,
            )
        ]

    assert any(
        "Ignoring unexpected tool execution event" in rec.message
        for rec in caplog.records
    )
    assert streamed[-1].type == StreamChunkType.TOOL_RESULT


def test_tool_result_chunk_for_message_preserves_metadata_and_logs_has_meta(caplog):
    """Test tool result chunk contains metadata enrichment and has_meta logging."""
    agent = _make_agent()
    caplog.set_level(logging.INFO)
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
        token_handler=TokenHandler(),
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
