"""Unit tests for DocsSummarizer PR2 class."""

import asyncio
import logging
from typing import ClassVar
from unittest.mock import ANY, AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel

from ols import config
from ols.app.models.config import LoggingConfig, MCPServerConfig
from ols.app.models.models import ChunkType

# needs to be setup before importing docs_summarizer
config.ols_config.authentication_config.module = "k8s"
from ols.src.query_helpers.docs_summarizer import (  # noqa: E402
    DocsSummarizer,
    QueryHelper,
    ToolTokenUsage,
)
from ols.src.tools.tools import ApprovalRequiredEvent, ToolResultEvent  # noqa: E402
from ols.utils.logging_configurator import configure_logging  # noqa: E402
from ols.utils.mcp_utils import build_mcp_config, gather_mcp_tools  # noqa: E402
from ols.utils.token_handler import TokenHandler  # noqa: E402
from tests import constants  # noqa: E402
from tests.mock_classes.mock_langchain_interface import (  # noqa: E402
    mock_langchain_interface,
)
from tests.mock_classes.mock_llm_loader import mock_llm_loader  # noqa: E402
from tests.mock_classes.mock_retrievers import MockRetriever  # noqa: E402
from tests.mock_classes.mock_tools import mock_tools_map  # noqa: E402


class SampleTool(StructuredTool):
    """Simple structured tool used for targeted docs_summarizer tests."""

    def __init__(self, name: str, description: str = "sample tool") -> None:
        """Initialize simple fake structured tool."""

        class _Schema(BaseModel):
            pass

        async def _coro(**kwargs):
            return "ok"

        super().__init__(
            name=name,
            description=description,
            func=lambda **kwargs: "ok",
            coroutine=_coro,
            args_schema=_Schema,
        )


def check_summary_result(summary, question: str) -> None:
    """Check result produced by DocsSummarizer.create_response method."""
    assert question in summary.response
    assert isinstance(summary.rag_chunks, list)
    assert len(summary.rag_chunks) == 1
    assert (
        f"{constants.OCP_DOCS_ROOT_URL}/{constants.OCP_DOCS_VERSION}/docs/test.html"
        in summary.rag_chunks[0].doc_url
    )
    assert summary.history_truncated is False
    assert summary.tool_calls == []
    assert summary.tool_results == []


@pytest.fixture(scope="function", autouse=True)
def _setup():
    """Set up config for tests."""
    config.reload_from_yaml_file("tests/config/valid_config_without_mcp.yaml")


def test_is_query_helper_subclass():
    """Test that DocsSummarizer is a subclass of QueryHelper."""
    assert issubclass(DocsSummarizer, QueryHelper)


def test_if_system_prompt_was_updated():
    """Test if system prompt was overridden from the configuration."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    assert summarizer._system_prompt == config.ols_config.system_prompt


def test_summarize_empty_history():
    """Basic test for DocsSummarizer using mocked retriever and empty history."""
    with (
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 1),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        summary = summarizer.create_response(question, MockRetriever(), [])
        check_summary_result(summary, question)


def test_summarize_no_history():
    """Basic test for DocsSummarizer without explicit history argument."""
    with (
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        summary = summarizer.create_response(question, MockRetriever())
        check_summary_result(summary, question)


def test_summarize_history_provided():
    """Basic test with explicit history vs default history paths."""
    with (
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        history = ["human: What is Kubernetes?"]
        rag_retriever = MockRetriever()

        with patch(
            "ols.src.query_helpers.docs_summarizer.TokenHandler.limit_conversation_history",
            return_value=([], False),
        ) as token_handler:
            summary1 = summarizer.create_response(question, rag_retriever, history)
            token_handler.assert_called_once_with(history, ANY)
            check_summary_result(summary1, question)

        with patch(
            "ols.src.query_helpers.docs_summarizer.TokenHandler.limit_conversation_history",
            return_value=([], False),
        ) as token_handler:
            summary2 = summarizer.create_response(question, rag_retriever)
            token_handler.assert_called_once_with([], ANY)
            check_summary_result(summary2, question)


def test_summarize_truncation():
    """Basic truncation check for very long history."""
    with patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summary = summarizer.create_response(
            "What's the ultimate question with answer 42?",
            MockRetriever(),
            [HumanMessage("What is Kubernetes?")] * 10000,
        )
        assert summary.history_truncated


def test_summarize_no_reference_content():
    """Basic test when no retriever is provided."""
    summarizer = DocsSummarizer(
        llm_loader=mock_llm_loader(mock_langchain_interface("test response")())
    )
    question = "What's the ultimate question with answer 42?"
    summary = summarizer.create_response(question)
    assert question in summary.response
    assert summary.rag_chunks == []
    assert not summary.history_truncated


def test_summarize_reranker(caplog):
    """Basic test to ensure reranker call is visible in logs."""
    logging_config = LoggingConfig(app_log_level="debug")
    configure_logging(logging_config)
    logger = logging.getLogger("ols")
    logger.handlers = [caplog.handler]

    with (
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        summary = summarizer.create_response(question, MockRetriever())
        check_summary_result(summary, question)
        assert "reranker.rerank() is called with 1 result(s)." in caplog.text


@pytest.mark.asyncio
async def test_response_generator():
    """Test response generator method."""
    summarizer = DocsSummarizer(
        llm_loader=mock_llm_loader(mock_langchain_interface("test response")())
    )
    question = "What's the ultimate question with answer 42?"
    generated_content = ""

    async for item in summarizer.generate_response(question):
        generated_content += item.text

    assert generated_content == question


async def async_mock_invoke(yield_values):
    """Mock async invoke_llm function to simulate LLM behavior."""
    for value in yield_values:
        yield value


def test_tool_calling_one_iteration():
    """Test tool calling - stops after one iteration."""
    with patch(
        "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm"
    ) as mock_invoke:
        mock_invoke.side_effect = lambda *args, **kwargs: async_mock_invoke(
            [AIMessageChunk(content="XYZ", response_metadata={"finish_reason": "stop"})]
        )
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tool_calling_enabled = True
        summarizer.create_response("How many namespaces are there in my cluster?")
        assert mock_invoke.call_count == 1


def test_tool_calling_drains_chunks_after_stop():
    """Test that chunks after finish_reason=stop are consumed but not forwarded."""
    question = "How many namespaces are there in my cluster?"

    with patch(
        "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm"
    ) as mock_invoke:
        mock_invoke.side_effect = lambda *args, **kwargs: async_mock_invoke(
            [
                AIMessageChunk(content="Hello", response_metadata={}),
                AIMessageChunk(content="", response_metadata={"finish_reason": "stop"}),
                AIMessageChunk(content="trailing1", response_metadata={}),
                AIMessageChunk(content="trailing2", response_metadata={}),
            ]
        )
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tool_calling_enabled = True
        summary = summarizer.create_response(question)
        assert mock_invoke.call_count == 1
        assert "Hello" in summary.response
        assert "trailing" not in summary.response


async def fake_invoke_llm(*args, **kwargs):
    """Fake invoke_llm function to simulate two-turn LLM behavior."""
    if not hasattr(fake_invoke_llm, "call_count"):
        fake_invoke_llm.call_count = 0
    fake_invoke_llm.call_count += 1

    if fake_invoke_llm.call_count == 1:
        yield AIMessageChunk(
            content="", response_metadata={"finish_reason": "tool_calls"}
        )
    elif fake_invoke_llm.call_count == 2:
        yield AIMessageChunk(content="XYZ", response_metadata={"finish_reason": "stop"})


def test_tool_calling_two_iteration():
    """Test tool calling - stops after two iterations."""
    with (
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm",
            new=fake_invoke_llm,
        ) as mock_invoke,
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new=AsyncMock(return_value=mock_tools_map),
        ),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tool_calling_enabled = True
        summarizer.create_response("How many namespaces are there in my cluster?")
        assert mock_invoke.call_count == 2


def test_tool_calling_force_stop():
    """Test tool calling - force stop by max rounds."""
    with (
        patch("ols.src.query_helpers.docs_summarizer.MAX_ITERATIONS", 3),
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm"
        ) as mock_invoke,
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new=AsyncMock(return_value=mock_tools_map),
        ),
    ):
        mock_invoke.side_effect = lambda *args, **kwargs: async_mock_invoke(
            [
                AIMessageChunk(
                    content="XYZ", response_metadata={"finish_reason": "tool_calls"}
                )
            ]
        )
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tool_calling_enabled = True
        summarizer.create_response("How many namespaces are there in my cluster?")
        assert mock_invoke.call_count == 3


def test_tool_calling_tool_execution(caplog):
    """Test tool execution path with one valid and one invalid tool call."""
    caplog.set_level(10)
    mcp_servers_config = {
        "test_server": {
            "transport": "streamable_http",
            "url": "http://test-server:8080/mcp",
        },
    }

    with (
        patch("ols.src.query_helpers.docs_summarizer.MAX_ITERATIONS", 2),
        patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_mcp_client_cls,
        patch(
            "ols.src.query_helpers.docs_summarizer.TokenHandler.calculate_and_check_available_tokens",
            return_value=1000,
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm"
        ) as mock_invoke,
        patch(
            "ols.src.query_helpers.docs_summarizer.build_mcp_config",
            return_value=mcp_servers_config,
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new=AsyncMock(return_value=mock_tools_map),
        ),
    ):
        mock_invoke.side_effect = lambda *args, **kwargs: async_mock_invoke(
            [
                AIMessageChunk(
                    content="",
                    response_metadata={"finish_reason": "tool_calls"},
                    tool_calls=[
                        {"name": "get_namespaces_mock", "args": {}, "id": "call_id1"},
                        {"name": "invalid_function_name", "args": {}, "id": "call_id2"},
                    ],
                )
            ]
        )

        mock_mcp_client_instance = AsyncMock()
        mock_mcp_client_instance.get_tools.return_value = mock_tools_map
        mock_mcp_client_cls.return_value = mock_mcp_client_instance

        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer.model_config.parameters.max_tokens_for_tools = 100
        summarizer.create_response("How many namespaces are there in my cluster?")

        assert "get_namespaces_mock" in caplog.text
        assert "invalid_function_name" in caplog.text
        assert mock_invoke.call_count == 2


@pytest.mark.asyncio
async def test_gather_mcp_tools_failure_isolation(caplog):
    """Test gather_mcp_tools isolates failures from individual MCP servers."""
    caplog.set_level(10)
    mcp_servers = {
        "working_server": {
            "transport": "streamable_http",
            "url": "http://working-server:8080/mcp",
        },
        "broken_server": {
            "transport": "streamable_http",
            "url": "http://non-exist:8888/mcp",
        },
    }

    async def mock_get_tools(server_name=None):
        if server_name == "working_server":
            return mock_tools_map
        raise ConnectionError("Failed to connect to http://non-exist:8888/mcp")

    with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
        mock_client_instance = AsyncMock()
        mock_client_instance.get_tools.side_effect = mock_get_tools
        mock_client_cls.return_value = mock_client_instance

        tools = await gather_mcp_tools(mcp_servers)
        assert len(tools) == 1
        assert tools[0].name == "get_namespaces_mock"
        assert "Loaded 1 tools from MCP server 'working_server'" in caplog.text
        assert "Failed to get tools from MCP server 'broken_server'" in caplog.text


def test_build_mcp_config_transport_is_streamable_http():
    """Test build_mcp_config sets transport to streamable_http for all servers."""
    server1 = MCPServerConfig(name="server1", url="http://server1:8080/mcp")
    server1._resolved_headers = {}
    server2 = MCPServerConfig(name="server2", url="http://server2:9090/mcp", timeout=30)
    server2._resolved_headers = {}

    mcp_config = build_mcp_config([server1, server2], None, None)

    assert mcp_config["server1"]["transport"] == "streamable_http"
    assert mcp_config["server2"]["transport"] == "streamable_http"


def test_resolve_tool_call_definitions_targeted_paths():
    """Test targeted paths in _resolve_tool_call_definitions helper."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    all_tools_dict = {"get_namespaces_mock": mock_tools_map[0]}
    duplicate_tool_names = {"dup_tool"}
    tool_calls = [
        {"name": None, "args": {}, "id": "missing_name"},
        {"name": "dup_tool", "args": {}, "id": "duplicate"},
        {"name": "not_found", "args": {}, "id": "unavailable"},
        {"name": "get_namespaces_mock", "args": "bad", "id": "bad_args"},
        {"name": "get_namespaces_mock", "args": {"ok": True}, "id": "valid"},
    ]

    definitions, skipped = summarizer._resolve_tool_call_definitions(
        tool_calls, all_tools_dict, duplicate_tool_names
    )

    assert len(definitions) == 1
    assert definitions[0][0] == "valid"
    assert definitions[0][1] == {"ok": True}
    assert definitions[0][2] is mock_tools_map[0]

    assert len(skipped) == 4
    skipped_ids = {msg.tool_call_id for msg in skipped}
    assert skipped_ids == {"missing_name", "duplicate", "unavailable", "bad_args"}


@pytest.mark.asyncio
async def test_collect_round_llm_chunks_targeted_paths():
    """Test _collect_round_llm_chunks returns chunks/text/stop as expected."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

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

    with patch.object(summarizer, "_invoke_llm", side_effect=_fake_invoke):
        tool_call_chunks, streamed_chunks, should_stop = (
            await summarizer._collect_round_llm_chunks(
                messages=[],
                llm_input_values={},
                all_mcp_tools=mock_tools_map,
                is_final_round=False,
                token_counter=AsyncMock(),
                round_index=1,
            )
        )

    assert should_stop is False
    assert len(streamed_chunks) == 1
    assert streamed_chunks[0].type == "text"
    assert streamed_chunks[0].text == "hello"
    assert len(tool_call_chunks) == 1


@pytest.mark.asyncio
async def test_process_tool_calls_for_round_streams_approval_and_result():
    """Test _process_tool_calls_for_round streams approval + tool_result and updates state."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
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
        "ols.src.query_helpers.docs_summarizer.execute_tool_calls_stream",
        side_effect=_fake_execute,
    ):
        streamed = [
            chunk
            async for chunk in summarizer._process_tool_calls_for_round(
                round_index=1,
                tool_call_chunks=tool_call_chunks,
                all_tools_dict={"get_namespaces_mock": mock_tools_map[0]},
                duplicate_tool_names=set(),
                messages=messages,
                token_handler=TokenHandler(),
                tool_token_usage=token_usage,
                max_tokens_for_tools=1000,
                max_tokens_per_tool=200,
            )
        ]

    assert [chunk.type for chunk in streamed] == [
        ChunkType.TOOL_CALL,
        ChunkType.APPROVAL_REQUIRED,
        ChunkType.TOOL_RESULT,
    ]
    assert streamed[1].data["approval_id"] == "aid-1"
    assert streamed[2].data["type"] == "tool_result"
    assert token_usage.used > 0
    # One AI tool-call message + one ToolMessage result appended into conversation state.
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_collect_round_llm_chunks_timeout_without_any_chunks():
    """Test round timeout path when LLM yields nothing before timeout."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

    async def _slow_invoke(*args, **kwargs):
        await asyncio.sleep(0.05)
        if False:
            yield AIMessageChunk(content="", response_metadata={})

    with (
        patch(
            "ols.src.query_helpers.docs_summarizer.constants.TOOL_CALL_ROUND_TIMEOUT",
            0.001,
        ),
        patch.object(summarizer, "_invoke_llm", side_effect=_slow_invoke),
    ):
        tool_call_chunks, streamed_chunks, should_stop = (
            await summarizer._collect_round_llm_chunks(
                messages=[],
                llm_input_values={},
                all_mcp_tools=mock_tools_map,
                is_final_round=False,
                token_counter=AsyncMock(),
                round_index=1,
            )
        )

    assert should_stop is True
    assert tool_call_chunks == []
    assert len(streamed_chunks) == 1
    assert streamed_chunks[0].type == ChunkType.TEXT
    assert "I could not complete this request in time." in streamed_chunks[0].text


@pytest.mark.asyncio
async def test_collect_round_llm_chunks_timeout_after_partial_text():
    """Test timeout still preserves already-streamed text before fallback."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

    async def _partial_then_slow(*args, **kwargs):
        yield AIMessageChunk(content="partial", response_metadata={})
        await asyncio.sleep(0.05)
        if False:
            yield AIMessageChunk(content="", response_metadata={})

    with (
        patch(
            "ols.src.query_helpers.docs_summarizer.constants.TOOL_CALL_ROUND_TIMEOUT",
            0.001,
        ),
        patch.object(summarizer, "_invoke_llm", side_effect=_partial_then_slow),
    ):
        tool_call_chunks, streamed_chunks, should_stop = (
            await summarizer._collect_round_llm_chunks(
                messages=[],
                llm_input_values={},
                all_mcp_tools=mock_tools_map,
                is_final_round=False,
                token_counter=AsyncMock(),
                round_index=1,
            )
        )

    assert should_stop is True
    assert tool_call_chunks == []
    assert [chunk.type for chunk in streamed_chunks] == [ChunkType.TEXT, ChunkType.TEXT]
    assert streamed_chunks[0].text == "partial"
    assert "I could not complete this request in time." in streamed_chunks[1].text


@pytest.mark.asyncio
async def test_collect_round_llm_chunks_stop_short_circuits_before_timeout():
    """Test finish_reason=stop returns immediately and does not emit timeout fallback."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

    async def _stop_immediately(*args, **kwargs):
        yield AIMessageChunk(content="", response_metadata={"finish_reason": "stop"})

    with (
        patch(
            "ols.src.query_helpers.docs_summarizer.constants.TOOL_CALL_ROUND_TIMEOUT",
            0.001,
        ),
        patch.object(summarizer, "_invoke_llm", side_effect=_stop_immediately),
    ):
        tool_call_chunks, streamed_chunks, should_stop = (
            await summarizer._collect_round_llm_chunks(
                messages=[],
                llm_input_values={},
                all_mcp_tools=mock_tools_map,
                is_final_round=False,
                token_counter=AsyncMock(),
                round_index=1,
            )
        )

    assert should_stop is True
    assert tool_call_chunks == []
    assert streamed_chunks == []


@pytest.mark.asyncio
async def test_collect_round_llm_chunks_handles_string_chunk():
    """Test fake-LLM compatibility path where chunk is plain string."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

    async def _string_invoke(*args, **kwargs):
        yield "plain-string-chunk"

    with patch.object(summarizer, "_invoke_llm", side_effect=_string_invoke):
        tool_call_chunks, streamed_chunks, should_stop = (
            await summarizer._collect_round_llm_chunks(
                messages=[],
                llm_input_values={},
                all_mcp_tools=[],
                is_final_round=False,
                token_counter=AsyncMock(),
                round_index=1,
            )
        )

    assert should_stop is False
    assert tool_call_chunks == []
    assert len(streamed_chunks) == 1
    assert streamed_chunks[0].type == "text"
    assert streamed_chunks[0].text == "plain-string-chunk"


def test_resolve_tool_call_definitions_none_args_normalized_to_empty_dict():
    """Test that None tool args are normalized to {}."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    tool = mock_tools_map[0]
    definitions, skipped = summarizer._resolve_tool_call_definitions(
        [{"name": tool.name, "args": None, "id": "call_none"}],
        {tool.name: tool},
        set(),
    )

    assert skipped == []
    assert len(definitions) == 1
    assert definitions[0][0] == "call_none"
    assert definitions[0][1] == {}
    assert definitions[0][2] is tool


@pytest.mark.asyncio
async def test_process_tool_calls_for_round_skipped_only_without_execution():
    """Test skipped-only path emits tool_result without calling executor."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
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
        "ols.src.query_helpers.docs_summarizer.execute_tool_calls_stream",
        new=AsyncMock(side_effect=AssertionError("executor should not be called")),
    ):
        streamed = [
            chunk
            async for chunk in summarizer._process_tool_calls_for_round(
                round_index=1,
                tool_call_chunks=tool_call_chunks,
                all_tools_dict={tool.name: tool},
                duplicate_tool_names=set(),
                messages=messages,
                token_handler=TokenHandler(),
                tool_token_usage=token_usage,
                max_tokens_for_tools=1000,
                max_tokens_per_tool=200,
            )
        ]

    assert [chunk.type for chunk in streamed] == [
        ChunkType.TOOL_CALL,
        ChunkType.TOOL_RESULT,
    ]
    assert streamed[1].data["type"] == "tool_result"
    assert "tool is unavailable" in streamed[1].data["content"]
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_process_tool_calls_for_round_ignores_unexpected_execution_event(caplog):
    """Test unexpected tool execution events are ignored with warning."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
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
        "ols.src.query_helpers.docs_summarizer.execute_tool_calls_stream",
        side_effect=_fake_execute,
    ):
        streamed = [
            chunk
            async for chunk in summarizer._process_tool_calls_for_round(
                round_index=1,
                tool_call_chunks=tool_call_chunks,
                all_tools_dict={"get_namespaces_mock": mock_tools_map[0]},
                duplicate_tool_names=set(),
                messages=messages,
                token_handler=TokenHandler(),
                tool_token_usage=token_usage,
                max_tokens_for_tools=1000,
                max_tokens_per_tool=200,
            )
        ]

    assert any(
        "Ignoring unexpected tool execution event" in rec.message
        for rec in caplog.records
    )
    assert streamed[-1].type == "tool_result"


@pytest.mark.asyncio
async def test_iterate_with_tools_deduplicates_tool_names(caplog):
    """Test duplicate MCP tool names are disabled and logged."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    summarizer._tool_calling_enabled = False
    caplog.set_level(logging.ERROR)
    tools = [SampleTool("dup"), SampleTool("dup")]

    with (
        patch.object(
            summarizer,
            "_collect_round_llm_chunks",
            new=AsyncMock(return_value=([], [], True)),
        ),
    ):
        chunks = [
            chunk
            async for chunk in summarizer.iterate_with_tools(
                messages=[],
                max_rounds=1,
                llm_input_values={},
                token_counter=AsyncMock(),
                all_mcp_tools=tools,
            )
        ]

    assert chunks == []
    assert "Duplicate MCP tool names detected and disabled" in caplog.text


def test_create_response_raises_on_unknown_chunk_type():
    """Test create_response raises ValueError on unsupported chunk type."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

    class UnknownChunk:
        type = "unsupported"
        text = ""
        data: ClassVar[dict[str, str]] = {}

    async def _fake_generate(self, *args, **kwargs):
        yield UnknownChunk()

    with patch.object(DocsSummarizer, "generate_response", _fake_generate):
        with pytest.raises(ValueError, match="Unknown chunk type"):
            summarizer.create_response("q")
