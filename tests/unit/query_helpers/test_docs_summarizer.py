"""Unit tests for DocsSummarizer PR2 class."""

import logging
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.ai import AIMessageChunk

from ols import config
from ols.app.models.config import LoggingConfig, MCPServerConfig, ModelParameters
from ols.app.models.models import StreamChunkType, StreamedChunk
from ols.constants import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_ITERATIONS_TROUBLESHOOTING,
    DEFAULT_TOOL_BUDGET_RATIO,
    DEFAULT_TOOL_BUDGET_RATIO_TROUBLESHOOTING,
    QueryMode,
)

# needs to be setup before importing docs_summarizer
config.ols_config.authentication_config.module = "k8s"

from ols.app.models.models import CacheEntry  # noqa: E402
from ols.src.query_helpers.docs_summarizer import (  # noqa: E402
    DocsSummarizer,
    QueryHelper,
)
from ols.utils.logging_configurator import configure_logging  # noqa: E402
from ols.utils.mcp_utils import build_mcp_config, gather_mcp_tools  # noqa: E402
from ols.utils.token_handler import (  # noqa: E402
    PromptTooLongError,
    TokenBudgetTracker,
    TokenCategory,
    TokenHandler,
)
from tests import constants  # noqa: E402
from tests.mock_classes.mock_langchain_interface import (  # noqa: E402
    mock_langchain_interface,
)
from tests.mock_classes.mock_llm_loader import mock_llm_loader  # noqa: E402
from tests.mock_classes.mock_retrievers import MockRetriever  # noqa: E402
from tests.mock_classes.mock_tools import mock_tools_map  # noqa: E402


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
        patch("ols.config.conversation_cache.get") as mock_cache_get,
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        rag_retriever = MockRetriever()

        mock_cache_get.return_value = [
            CacheEntry(query=HumanMessage("What is Kubernetes?"))
        ]
        with patch(
            "ols.src.query_helpers.docs_summarizer.TokenHandler.limit_conversation_history",
            return_value=([], False),
        ) as token_handler:
            summary1 = summarizer.create_response(
                question, rag_retriever, "user-id", "conv-id"
            )
            # Non-overflow path returns early from prepare_history (no second limit pass).
            token_handler.assert_not_called()
            check_summary_result(summary1, "What is Kubernetes?")

        mock_cache_get.return_value = []
        with patch(
            "ols.src.query_helpers.docs_summarizer.TokenHandler.limit_conversation_history",
            return_value=([], False),
        ) as token_handler:
            summary2 = summarizer.create_response(
                question, rag_retriever, "user-id", "conv-id2"
            )
            token_handler.assert_not_called()
            check_summary_result(summary2, question)


def test_summarize_truncation():
    """Basic test for DocsSummarizer to check compression avoids truncation."""
    with (
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.config.conversation_cache.get") as mock_cache_get,
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        rag_retriever = MockRetriever()

        history = [
            CacheEntry(
                query=HumanMessage("What is Kubernetes?" * 100),
                response=AIMessage(
                    "Kubernetes is a container orchestration system." * 100
                ),
            )
        ] * 100
        mock_cache_get.return_value = history

        summary = summarizer.create_response(
            question, rag_retriever, "user-id", "conv-id"
        )

        assert not summary.history_truncated


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


def test_summarize_retrieval_logging(caplog):
    """Basic test to ensure retrieval details are visible in logs."""
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
        assert "Retrieved 1 documents from indexes" in caplog.text


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


@pytest.mark.asyncio
async def test_response_generator_emits_history_events_before_tokens():
    """Test history compression events are emitted in order before text chunks."""
    summarizer = DocsSummarizer(
        llm_loader=mock_llm_loader(mock_langchain_interface("test response")())
    )
    question = "What's the ultimate question with answer 42?"

    async def mock_prepare_history(**kwargs):
        yield StreamedChunk(
            type=StreamChunkType.HISTORY_COMPRESSION_START,
            data={"status": "started"},
        )
        yield StreamedChunk(
            type=StreamChunkType.HISTORY_COMPRESSION_END,
            data={"status": "completed", "duration_ms": 1.0},
        )
        yield ([], False)

    chunk_types: list[StreamChunkType] = []

    with patch(
        "ols.src.query_helpers.docs_summarizer.prepare_history",
        side_effect=mock_prepare_history,
    ):
        chunk_types.extend(
            [item.type async for item in summarizer.generate_response(question)]
        )

    assert StreamChunkType.HISTORY_COMPRESSION_START in chunk_types
    assert StreamChunkType.HISTORY_COMPRESSION_END in chunk_types
    assert StreamChunkType.TEXT in chunk_types
    assert chunk_types.index(
        StreamChunkType.HISTORY_COMPRESSION_START
    ) < chunk_types.index(StreamChunkType.HISTORY_COMPRESSION_END)
    assert chunk_types.index(
        StreamChunkType.HISTORY_COMPRESSION_END
    ) < chunk_types.index(StreamChunkType.TEXT)


async def async_mock_invoke(yield_values):
    """Mock async invoke_llm function to simulate LLM behavior."""
    for value in yield_values:
        yield value


def test_tool_calling_one_iteration():
    """Test tool calling - stops after one iteration."""
    with patch(
        "ols.src.query_helpers.llm_execution_agent.LLMExecutionAgent._invoke_llm"
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
        "ols.src.query_helpers.llm_execution_agent.LLMExecutionAgent._invoke_llm"
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
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_call_chunks=[
                {
                    "name": "get_namespaces_mock",
                    "args": "{}",
                    "id": "call_1",
                    "index": 0,
                }
            ],
        )
    elif fake_invoke_llm.call_count == 2:
        yield AIMessageChunk(content="XYZ", response_metadata={"finish_reason": "stop"})


def test_tool_calling_two_iteration():
    """Test tool calling - stops after two iterations."""
    with (
        patch(
            "ols.src.query_helpers.llm_execution_agent.LLMExecutionAgent._invoke_llm",
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
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._get_max_iterations",
            return_value=3,
        ),
        patch(
            "ols.src.query_helpers.llm_execution_agent.LLMExecutionAgent._invoke_llm"
        ) as mock_invoke,
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
                    tool_call_chunks=[
                        {
                            "name": "get_namespaces_mock",
                            "args": "{}",
                            "id": "call_1",
                            "index": 0,
                        }
                    ],
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
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._get_max_iterations",
            return_value=2,
        ),
        patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_mcp_client_cls,
        patch(
            "ols.src.query_helpers.docs_summarizer.TokenBudgetTracker.tools_round_budget",
            new_callable=PropertyMock,
            return_value=1000,
        ),
        patch(
            "ols.src.query_helpers.llm_execution_agent.LLMExecutionAgent._invoke_llm"
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
        summarizer.model_config.max_tokens_for_tools = 100
        summarizer.create_response("How many namespaces are there in my cluster?")

        assert "get_namespaces_mock" in caplog.text
        assert "invalid_function_name" in caplog.text
        assert mock_invoke.call_count == 2


def test_build_final_prompt_raises_when_tool_definitions_exceed_prompt_budget() -> None:
    """Tool definitions token estimate is validated with the final prompt budget."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    summarizer._tracker = TokenBudgetTracker(
        token_handler=TokenHandler(),
        context_window_size=1000,
        max_response_tokens=100,
        max_tool_tokens=200,
        round_cap_fraction=0.6,
    )
    summarizer._tracker.set_tool_loop_max_rounds(5)
    summarizer._tracker.charge(TokenCategory.PROMPT, 650)
    with pytest.raises(PromptTooLongError, match="Tool definitions"):
        summarizer._build_final_prompt(
            query="q",
            history=[],
            rag_chunks=[],
            skill_content=None,
            tool_definitions_tokens=100,
        )


def test_tool_output_token_tracking_uses_buffer_weight(caplog):
    """Test that tool output tokens are counted with TOKEN_BUFFER_WEIGHT like other budget items.

    Before this fix, raw len(tokens) was used for tool outputs while tool definitions
    and AIMessage tokens used _get_token_count() (which applies a 1.1x buffer).
    This test asserts _get_token_count() is called for tool output tokens by spying on
    it: with one tool call in one round it must be called at least 3 times
    (tool definitions, AIMessage, tool output).
    """
    mcp_servers_config = {
        "test_server": {
            "transport": "streamable_http",
            "url": "http://test-server:8080/mcp",
        },
    }

    original_get_token_count = TokenHandler._get_token_count
    call_count = 0

    def counting_get_token_count(tokens: list) -> int:
        nonlocal call_count
        call_count += 1
        return original_get_token_count(tokens)

    with (
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._get_max_iterations",
            return_value=2,
        ),
        patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_mcp_client_cls,
        patch(
            "ols.src.query_helpers.llm_execution_agent.LLMExecutionAgent._invoke_llm"
        ) as mock_invoke,
        patch("ols.utils.mcp_utils.config") as mock_config,
        patch.object(
            TokenHandler, "_get_token_count", staticmethod(counting_get_token_count)
        ),
    ):
        mock_config.tools_rag = None
        mock_config.mcp_servers.servers = [MagicMock()]

        with patch(
            "ols.utils.mcp_utils._gather_and_populate_tools",
            new=AsyncMock(return_value=(mcp_servers_config, mock_tools_map)),
        ):
            mock_invoke.side_effect = lambda *args, **kwargs: async_mock_invoke(
                [
                    AIMessageChunk(
                        content="",
                        response_metadata={"finish_reason": "tool_calls"},
                        tool_calls=[
                            {
                                "name": "get_namespaces_mock",
                                "args": {},
                                "id": "call_id1",
                            },
                        ],
                    )
                ]
            )

        mock_mcp_client_instance = AsyncMock()
        mock_mcp_client_instance.get_tools.return_value = mock_tools_map
        mock_mcp_client_cls.return_value = mock_mcp_client_instance

        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer.model_config.max_tokens_for_tools = 50000
        summarizer.create_response("How many namespaces?")

    # _get_token_count must be called for:
    #   1. tool definitions (once at the start of the loop)
    #   2. AIMessage with tool_calls
    #   3. tool output (the change introduced by this fix)
    assert call_count >= 3, (
        f"Expected _get_token_count to be called at least 3 times "
        f"(definitions + AIMessage + tool output), got {call_count}"
    )


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


def test_get_max_iterations_ask_mode_no_override():
    """Test _get_max_iterations returns ASK default when config has no override."""
    config.ols_config.max_iterations = None
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None), mode=QueryMode.ASK)
    assert summarizer._get_max_iterations() == DEFAULT_MAX_ITERATIONS


def test_get_max_iterations_troubleshooting_mode_no_override():
    """Test _get_max_iterations returns TROUBLESHOOTING default when config has no override."""
    config.ols_config.max_iterations = None
    summarizer = DocsSummarizer(
        llm_loader=mock_llm_loader(None), mode=QueryMode.TROUBLESHOOTING
    )
    assert summarizer._get_max_iterations() == DEFAULT_MAX_ITERATIONS_TROUBLESHOOTING


def test_get_max_iterations_config_override_above_default():
    """Test _get_max_iterations uses config value when it exceeds the mode default."""
    config.ols_config.max_iterations = 20
    try:
        for mode in (QueryMode.ASK, QueryMode.TROUBLESHOOTING):
            summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None), mode=mode)
            assert summarizer._get_max_iterations() == 20
    finally:
        config.ols_config.max_iterations = None


def test_get_max_iterations_config_override_below_default():
    """Test _get_max_iterations uses mode default when it exceeds the config value."""
    config.ols_config.max_iterations = 10
    try:
        summarizer = DocsSummarizer(
            llm_loader=mock_llm_loader(None), mode=QueryMode.ASK
        )
        assert summarizer._get_max_iterations() == 10

        summarizer = DocsSummarizer(
            llm_loader=mock_llm_loader(None), mode=QueryMode.TROUBLESHOOTING
        )
        assert (
            summarizer._get_max_iterations() == DEFAULT_MAX_ITERATIONS_TROUBLESHOOTING
        )
    finally:
        config.ols_config.max_iterations = None


def _default_m1_model():
    """Return the default model config used by DocsSummarizer tests."""
    return config.llm_config.providers["p1"].models["m1"]


def test_tracker_max_tool_tokens_troubleshooting_uses_mode_floor():
    """Troubleshooting mode raises tool budget to the mode floor over default ratio."""
    m1 = _default_m1_model()
    saved = m1.parameters
    m1.parameters = ModelParameters(max_tokens_for_response=100)
    mcp_servers_config = {
        "test_server": {
            "transport": "streamable_http",
            "url": "http://test-server:8080/mcp",
        },
    }
    try:
        with patch(
            "ols.src.query_helpers.docs_summarizer.build_mcp_config",
            return_value=mcp_servers_config,
        ):
            summarizer = DocsSummarizer(
                llm_loader=mock_llm_loader(None), mode=QueryMode.TROUBLESHOOTING
            )
        cw = summarizer.model_config.context_window_size
        assert summarizer._tracker.max_tool_tokens == int(
            cw * DEFAULT_TOOL_BUDGET_RATIO_TROUBLESHOOTING
        )
    finally:
        m1.parameters = saved


def test_tracker_max_tool_tokens_ask_mode_uses_default_ratio():
    """Ask mode keeps the default tool budget ratio."""
    m1 = _default_m1_model()
    saved = m1.parameters
    m1.parameters = ModelParameters(max_tokens_for_response=100)
    mcp_servers_config = {
        "test_server": {
            "transport": "streamable_http",
            "url": "http://test-server:8080/mcp",
        },
    }
    try:
        with patch(
            "ols.src.query_helpers.docs_summarizer.build_mcp_config",
            return_value=mcp_servers_config,
        ):
            summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None), mode=QueryMode.ASK)
        cw = summarizer.model_config.context_window_size
        assert summarizer._tracker.max_tool_tokens == int(cw * DEFAULT_TOOL_BUDGET_RATIO)
    finally:
        m1.parameters = saved


def test_tracker_max_tool_tokens_troubleshooting_explicit_ratio_above_floor():
    """Configured tool_budget_ratio above the mode floor is retained."""
    m1 = _default_m1_model()
    saved = m1.parameters
    m1.parameters = ModelParameters(max_tokens_for_response=100, tool_budget_ratio=0.55)
    mcp_servers_config = {
        "test_server": {
            "transport": "streamable_http",
            "url": "http://test-server:8080/mcp",
        },
    }
    try:
        with patch(
            "ols.src.query_helpers.docs_summarizer.build_mcp_config",
            return_value=mcp_servers_config,
        ):
            summarizer = DocsSummarizer(
                llm_loader=mock_llm_loader(None), mode=QueryMode.TROUBLESHOOTING
            )
        cw = summarizer.model_config.context_window_size
        assert summarizer._tracker.max_tool_tokens == int(cw * 0.55)
    finally:
        m1.parameters = saved


def test_tracker_max_tool_tokens_troubleshooting_explicit_ratio_below_floor():
    """Configured tool_budget_ratio below the troubleshooting floor is raised."""
    m1 = _default_m1_model()
    saved = m1.parameters
    m1.parameters = ModelParameters(max_tokens_for_response=100, tool_budget_ratio=0.3)
    mcp_servers_config = {
        "test_server": {
            "transport": "streamable_http",
            "url": "http://test-server:8080/mcp",
        },
    }
    try:
        with patch(
            "ols.src.query_helpers.docs_summarizer.build_mcp_config",
            return_value=mcp_servers_config,
        ):
            summarizer = DocsSummarizer(
                llm_loader=mock_llm_loader(None), mode=QueryMode.TROUBLESHOOTING
            )
        cw = summarizer.model_config.context_window_size
        assert summarizer._tracker.max_tool_tokens == int(
            cw * DEFAULT_TOOL_BUDGET_RATIO_TROUBLESHOOTING
        )
    finally:
        m1.parameters = saved


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


def test_create_response_ignores_reasoning_chunks():
    """Test create_response skips reasoning chunks without error."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    from ols.app.models.models import StreamedChunk

    async def _fake_generate(self, *args, **kwargs):
        yield StreamedChunk(type=StreamChunkType.REASONING, text="thinking")
        yield StreamedChunk(type=StreamChunkType.TEXT, text="answer")
        yield StreamedChunk(
            type=StreamChunkType.END,
            data={"rag_chunks": [], "truncated": False, "token_counter": None},
        )

    with patch.object(DocsSummarizer, "generate_response", _fake_generate):
        result = summarizer.create_response("q")

    assert result.response == "answer"
