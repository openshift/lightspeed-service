"""Unit tests for DocsSummarizer class."""

import logging
from unittest.mock import ANY, AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessageChunk

from ols import config
from tests.mock_classes.mock_tools import mock_tools_map

# needs to be setup there before is_user_authorized is imported
config.ols_config.authentication_config.module = "k8s"


from ols.app.models.config import (  # noqa:E402
    LoggingConfig,
)
from ols.src.query_helpers.docs_summarizer import (  # noqa:E402
    DocsSummarizer,
    QueryHelper,
)
from ols.utils import suid  # noqa:E402
from ols.utils.logging_configurator import configure_logging  # noqa:E402
from tests import constants  # noqa:E402
from tests.mock_classes.mock_langchain_interface import (  # noqa:E402
    mock_langchain_interface,
)
from tests.mock_classes.mock_llm_loader import mock_llm_loader  # noqa:E402
from tests.mock_classes.mock_retrievers import MockRetriever  # noqa:E402

conversation_id = suid.get_suid()


def test_is_query_helper_subclass():
    """Test that DocsSummarizer is a subclass of QueryHelper."""
    assert issubclass(DocsSummarizer, QueryHelper)


def check_summary_result(summary, question):
    """Check result produced by DocsSummarizer.summary method."""
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


def test_if_system_prompt_was_updated():
    """Test if system prompt was overided from the configuration."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    # expected prompt was loaded during configuration phase
    expected_prompt = config.ols_config.system_prompt
    assert summarizer._system_prompt == expected_prompt


def test_summarize_empty_history():
    """Basic test for DocsSummarizer using mocked index and query engine."""
    with (
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 1),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        rag_retriever = MockRetriever()
        history = []  # empty history
        summary = summarizer.create_response(question, rag_retriever, history)
        check_summary_result(summary, question)


def test_summarize_no_history():
    """Basic test for DocsSummarizer using mocked index and query engine, no history is provided."""
    with (
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        rag_retriever = MockRetriever()
        # no history is passed into summarize() method
        summary = summarizer.create_response(question, rag_retriever)
        check_summary_result(summary, question)


def test_summarize_history_provided():
    """Basic test for DocsSummarizer using mocked index and query engine, history is provided."""
    with (
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        history = ["human: What is Kubernetes?"]
        rag_retriever = MockRetriever()

        # first call with history provided
        with patch(
            "ols.src.query_helpers.docs_summarizer.TokenHandler.limit_conversation_history",
            return_value=([], False),
        ) as token_handler:
            summary1 = summarizer.create_response(question, rag_retriever, history)
            token_handler.assert_called_once_with(history, ANY)
            check_summary_result(summary1, question)

        # second call without history provided
        with patch(
            "ols.src.query_helpers.docs_summarizer.TokenHandler.limit_conversation_history",
            return_value=([], False),
        ) as token_handler:
            summary2 = summarizer.create_response(question, rag_retriever)
            token_handler.assert_called_once_with([], ANY)
            check_summary_result(summary2, question)


def test_summarize_truncation():
    """Basic test for DocsSummarizer to check if truncation is done."""
    with patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        rag_retriever = MockRetriever()

        # too long history
        history = [HumanMessage("What is Kubernetes?")] * 10000
        summary = summarizer.create_response(question, rag_retriever, history)

        # truncation should be done
        assert summary.history_truncated


def test_summarize_no_reference_content():
    """Basic test for DocsSummarizer using mocked index and query engine."""
    summarizer = DocsSummarizer(
        llm_loader=mock_llm_loader(mock_langchain_interface("test response")())
    )
    question = "What's the ultimate question with answer 42?"
    summary = summarizer.create_response(question)
    assert question in summary.response
    assert summary.rag_chunks == []
    assert not summary.history_truncated


def test_summarize_reranker(caplog):
    """Basic test to make sure the reranker is called as expected."""
    logging_config = LoggingConfig(app_log_level="debug")

    configure_logging(logging_config)
    logger = logging.getLogger("ols")
    logger.handlers = [caplog.handler]  # add caplog handler to logger

    with (
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        rag_retriever = MockRetriever()
        # no history is passed into create_response() method
        summary = summarizer.create_response(question, rag_retriever)
        check_summary_result(summary, question)

        # Check captured log text to see if reranker was called.
        assert "reranker.rerank() is called with 1 result(s)." in caplog.text


@pytest.mark.asyncio
async def test_response_generator():
    """Test response generator method."""
    summarizer = DocsSummarizer(
        llm_loader=mock_llm_loader(mock_langchain_interface("test response")())
    )
    question = "What's the ultimate question with answer 42?"
    summary_gen = summarizer.generate_response(question)
    generated_content = ""

    async for item in summary_gen:
        generated_content += item.text

    assert generated_content == question


async def async_mock_invoke(yield_values):
    """Mock async invoke_llm function to simulate LLM behavior."""
    for value in yield_values:
        yield value


def test_tool_calling_one_iteration():
    """Test tool calling - stops after one iteration."""
    question = "How many namespaces are there in my cluster?"

    with patch(
        "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm"
    ) as mock_invoke:
        mock_invoke.side_effect = lambda *args, **kwargs: async_mock_invoke(
            [AIMessageChunk(content="XYZ", response_metadata={"finish_reason": "stop"})]
        )
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tool_calling_enabled = True
        summarizer.create_response(question)
        assert mock_invoke.call_count == 1


async def fake_invoke_llm(*args, **kwargs):
    """Fake invoke_llm function to simulate LLM behavior.

    Yields depends on the number of calls
    """
    # use an attribute on the function to track calls
    if not hasattr(fake_invoke_llm, "call_count"):
        fake_invoke_llm.call_count = 0
    fake_invoke_llm.call_count += 1

    if fake_invoke_llm.call_count == 1:
        # first call yields a message that requests tool calls
        yield AIMessageChunk(
            content="", response_metadata={"finish_reason": "tool_calls"}
        )
    elif fake_invoke_llm.call_count == 2:
        # second call yields the final message.
        yield AIMessageChunk(content="XYZ", response_metadata={"finish_reason": "stop"})
    else:
        # extra
        yield AIMessageChunk(
            content="Extra", response_metadata={"finish_reason": "extra"}
        )


def test_tool_calling_two_iteration():
    """Test tool calling - stops after two iterations."""
    question = "How many namespaces are there in my cluster?"

    with patch(
        "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm",
        new=fake_invoke_llm,
    ) as mock_invoke:
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tool_calling_enabled = True
        summarizer.create_response(question)
        assert mock_invoke.call_count == 2


def test_tool_calling_force_stop():
    """Test tool calling - force stop."""
    question = "How many namespaces are there in my cluster?"

    with (
        patch("ols.src.query_helpers.docs_summarizer.MAX_ITERATIONS", 3),
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm"
        ) as mock_invoke,
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
        summarizer.create_response(question)
        assert mock_invoke.call_count == 3


def test_tool_calling_tool_execution(caplog):
    """Test tool calling - tool execution."""
    caplog.set_level(10)  # Set debug level

    question = "How many namespaces are there in my cluster?"

    mcp_servers_config = {
        "test_server": {
            "transport": "streamable_http",
            "url": "http://test-server:8080/mcp",
        },
    }

    with (
        patch("ols.src.query_helpers.docs_summarizer.MAX_ITERATIONS", 2),
        patch(
            "ols.src.query_helpers.docs_summarizer.MultiServerMCPClient"
        ) as mock_mcp_client_cls,
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm"
        ) as mock_invoke,
        patch(
            "ols.src.query_helpers.docs_summarizer.MCPConfigBuilder.dump_client_config",
            return_value=mcp_servers_config,
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

        # Create mock MCP client - now get_tools is called with server_name parameter
        mock_mcp_client_instance = AsyncMock()
        mock_mcp_client_instance.get_tools.return_value = mock_tools_map
        mock_mcp_client_cls.return_value = mock_mcp_client_instance

        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        # Disable token reservation for tools in this test (test config has small context window)
        summarizer.model_config.parameters.max_tokens_for_tools = 0
        summarizer.create_response(question)

        assert "Tool: get_namespaces_mock" in caplog.text
        tool_output = mock_tools_map[0].invoke({})
        assert f"Output: {tool_output}" in caplog.text

        assert "Error: Tool 'invalid_function_name' not found." in caplog.text

        assert mock_invoke.call_count == 2


@pytest.mark.asyncio
async def test_gather_mcp_tools_failure_isolation(caplog):
    """Test gather_mcp_tools isolates failures from individual MCP servers.

    When multiple MCP servers are configured and one is unreachable,
    tools from the working servers should still be returned.
    """
    from ols.src.query_helpers.docs_summarizer import gather_mcp_tools

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

    # Mock MultiServerMCPClient.get_tools to simulate per-server behavior
    async def mock_get_tools(server_name=None):
        if server_name == "working_server":
            return mock_tools_map
        elif server_name == "broken_server":
            raise ConnectionError("Failed to connect to http://non-exist:8888/mcp")
        return []

    with patch(
        "ols.src.query_helpers.docs_summarizer.MultiServerMCPClient"
    ) as mock_client_cls:
        mock_client_instance = AsyncMock()
        mock_client_instance.get_tools.side_effect = mock_get_tools
        mock_client_cls.return_value = mock_client_instance

        # Call gather_mcp_tools - should return tools from working server
        # even though broken_server fails
        tools = await gather_mcp_tools(mcp_servers)

        # Verify we got tools from the working server
        assert len(tools) == 1
        assert tools[0].name == "get_namespaces_mock"

        # Verify logging shows partial success
        assert "Loaded 1 tools from MCP server 'working_server'" in caplog.text
        assert "Failed to get tools from MCP server 'broken_server'" in caplog.text


@pytest.mark.asyncio
async def test_gather_mcp_tools_all_servers_working(caplog):
    """Test gather_mcp_tools aggregates tools from all working servers."""
    from ols.src.query_helpers.docs_summarizer import gather_mcp_tools

    caplog.set_level(10)

    mcp_servers = {
        "server_a": {"transport": "streamable_http", "url": "http://server-a:8080/mcp"},
        "server_b": {"transport": "streamable_http", "url": "http://server-b:8080/mcp"},
    }

    async def mock_get_tools(server_name=None):
        # Both servers return tools successfully
        return mock_tools_map

    with patch(
        "ols.src.query_helpers.docs_summarizer.MultiServerMCPClient"
    ) as mock_client_cls:
        mock_client_instance = AsyncMock()
        mock_client_instance.get_tools.side_effect = mock_get_tools
        mock_client_cls.return_value = mock_client_instance

        tools = await gather_mcp_tools(mcp_servers)

        # Should have tools from both servers (2 x 1 = 2 tools)
        assert len(tools) == 2
        assert "Loaded 1 tools from MCP server 'server_a'" in caplog.text
        assert "Loaded 1 tools from MCP server 'server_b'" in caplog.text


@pytest.mark.asyncio
async def test_gather_mcp_tools_all_servers_failing(caplog):
    """Test gather_mcp_tools handles all servers failing gracefully."""
    from ols.src.query_helpers.docs_summarizer import gather_mcp_tools

    caplog.set_level(10)

    mcp_servers = {
        "broken_a": {"transport": "streamable_http", "url": "http://broken-a:8888/mcp"},
        "broken_b": {"transport": "streamable_http", "url": "http://broken-b:8888/mcp"},
    }

    async def mock_get_tools(server_name=None):
        raise ConnectionError(f"Failed to connect to {server_name}")

    with patch(
        "ols.src.query_helpers.docs_summarizer.MultiServerMCPClient"
    ) as mock_client_cls:
        mock_client_instance = AsyncMock()
        mock_client_instance.get_tools.side_effect = mock_get_tools
        mock_client_cls.return_value = mock_client_instance

        tools = await gather_mcp_tools(mcp_servers)

        # Should return empty list, not raise exception
        assert tools == []
        assert "Failed to get tools from MCP server 'broken_a'" in caplog.text
        assert "Failed to get tools from MCP server 'broken_b'" in caplog.text


@pytest.mark.asyncio
async def test_gather_mcp_tools_empty_config():
    """Test gather_mcp_tools with no servers configured."""
    from ols.src.query_helpers.docs_summarizer import gather_mcp_tools

    with patch(
        "ols.src.query_helpers.docs_summarizer.MultiServerMCPClient"
    ) as mock_client_cls:
        mock_client_instance = AsyncMock()
        mock_client_cls.return_value = mock_client_instance

        tools = await gather_mcp_tools({})

        # Should return empty list
        assert tools == []
        # get_tools should never be called
        mock_client_instance.get_tools.assert_not_called()
