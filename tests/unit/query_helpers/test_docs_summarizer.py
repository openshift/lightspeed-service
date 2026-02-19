"""Unit tests for DocsSummarizer class."""

import json
import logging
import re
from math import ceil
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessageChunk

from ols import config
from ols.app.models.config import MCPServerConfig
from ols.constants import TOKEN_BUFFER_WEIGHT
from ols.utils.token_handler import TokenHandler
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

    with (
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm",
            new=fake_invoke_llm,
        ) as mock_invoke,
        patch("ols.utils.mcp_utils.config") as mock_config,
    ):
        # Mock config for get_mcp_tools
        mock_config.tools_rag = None
        mock_config.mcp_servers.servers = [MagicMock()]  # Non-empty list

        # Mock _gather_and_populate_tools to return tools
        with patch(
            "ols.utils.mcp_utils._gather_and_populate_tools",
            new=AsyncMock(return_value=({"test": {}}, mock_tools_map)),
        ):
            summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
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
        patch("ols.utils.mcp_utils.config") as mock_config,
    ):
        # Mock config for get_mcp_tools
        mock_config.tools_rag = None
        mock_config.mcp_servers.servers = [MagicMock()]  # Non-empty list

        # Mock _gather_and_populate_tools to return tools
        with patch(
            "ols.utils.mcp_utils._gather_and_populate_tools",
            new=AsyncMock(return_value=({"test": {}}, mock_tools_map)),
        ):
            mock_invoke.side_effect = lambda *args, **kwargs: async_mock_invoke(
                [
                    AIMessageChunk(
                        content="XYZ", response_metadata={"finish_reason": "tool_calls"}
                    )
                ]
            )
            summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
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
        patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_mcp_client_cls,
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm"
        ) as mock_invoke,
        patch("ols.utils.mcp_utils.config") as mock_config,
    ):
        # Mock config for get_mcp_tools
        mock_config.tools_rag = None
        mock_config.mcp_servers.servers = [MagicMock()]  # Non-empty list

        # Mock _gather_and_populate_tools to return tools
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
                            {
                                "name": "invalid_function_name",
                                "args": {},
                                "id": "call_id2",
                            },
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


def test_tool_token_tracking(caplog):
    """Test that tool definitions and AIMessage tokens are tracked with buffer weight."""
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
        patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_mcp_client_cls,
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm"
        ) as mock_invoke,
        patch("ols.utils.mcp_utils.config") as mock_config,
    ):
        # Mock config for get_mcp_tools
        mock_config.tools_rag = None
        mock_config.mcp_servers.servers = [MagicMock()]  # Non-empty list

        # Mock _gather_and_populate_tools to return tools
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
        # Disable token reservation for tools (test config has small context window)
        summarizer.model_config.parameters.max_tokens_for_tools = 0
        summarizer.create_response(question)

        # Verify tool definitions token counting is logged
        assert "Tool definitions consume" in caplog.text

        # Calculate expected token count with buffer weight applied
        token_handler = TokenHandler()
        tool_definitions_text = json.dumps(
            [
                {"name": t.name, "description": t.description, "schema": t.args}
                for t in mock_tools_map
            ]
        )
        raw_tokens = len(token_handler.text_to_tokens(tool_definitions_text))
        expected_buffered_tokens = ceil(raw_tokens * TOKEN_BUFFER_WEIGHT)

        # Extract logged token count and verify buffer weight was applied
        match = re.search(r"Tool definitions consume (\d+) tokens", caplog.text)
        assert match is not None, "Token count not found in logs"
        logged_tokens = int(match.group(1))
        assert logged_tokens == expected_buffered_tokens, (
            f"Expected {expected_buffered_tokens} (raw={raw_tokens} * {TOKEN_BUFFER_WEIGHT}), "
            f"got {logged_tokens}"
        )


@pytest.mark.asyncio
async def test_gather_mcp_tools_failure_isolation(caplog):
    """Test gather_mcp_tools isolates failures from individual MCP servers.

    When multiple MCP servers are configured and one is unreachable,
    tools from the working servers should still be returned.
    """
    from ols.utils.mcp_utils import gather_mcp_tools

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

    with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
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
    from ols.utils.mcp_utils import gather_mcp_tools

    caplog.set_level(10)

    mcp_servers = {
        "server_a": {"transport": "streamable_http", "url": "http://server-a:8080/mcp"},
        "server_b": {"transport": "streamable_http", "url": "http://server-b:8080/mcp"},
    }

    async def mock_get_tools(server_name=None):
        # Both servers return tools successfully
        return mock_tools_map

    with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
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
    from ols.utils.mcp_utils import gather_mcp_tools

    caplog.set_level(10)

    mcp_servers = {
        "broken_a": {"transport": "streamable_http", "url": "http://broken-a:8888/mcp"},
        "broken_b": {"transport": "streamable_http", "url": "http://broken-b:8888/mcp"},
    }

    async def mock_get_tools(server_name=None):
        raise ConnectionError(f"Failed to connect to {server_name}")

    with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
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
    from ols.utils.mcp_utils import gather_mcp_tools

    with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
        mock_client_instance = AsyncMock()
        mock_client_cls.return_value = mock_client_instance

        tools = await gather_mcp_tools({})

        # Should return empty list
        assert tools == []
        # get_tools should never be called
        mock_client_instance.get_tools.assert_not_called()


def test_build_mcp_config_transport_is_streamable_http():
    """Test build_mcp_config sets transport to streamable_http for all servers."""
    from ols.utils.mcp_utils import build_mcp_config

    server1 = MCPServerConfig(name="server1", url="http://server1:8080/mcp")
    server1._resolved_headers = {}

    server2 = MCPServerConfig(name="server2", url="http://server2:9090/mcp", timeout=30)
    server2._resolved_headers = {}

    mcp_config = build_mcp_config(
        [server1, server2], user_token=None, client_headers=None
    )

    assert "server1" in mcp_config
    assert "server2" in mcp_config

    assert mcp_config["server1"]["transport"] == "streamable_http"
    assert mcp_config["server1"]["url"] == "http://server1:8080/mcp"

    assert mcp_config["server2"]["transport"] == "streamable_http"
    assert mcp_config["server2"]["url"] == "http://server2:9090/mcp"
    assert mcp_config["server2"]["timeout"] == 30


def test_resolve_server_headers_with_client_placeholder():
    """Test resolve_server_headers replaces client placeholder with client headers."""
    from ols.constants import MCP_CLIENT_PLACEHOLDER
    from ols.utils.mcp_utils import resolve_server_headers

    server = MCPServerConfig(
        name="test-server",
        url="http://test:8080/mcp",
        headers={"Authorization": "_client_"},
    )
    server._resolved_headers = {"Authorization": MCP_CLIENT_PLACEHOLDER}

    client_headers = {"test-server": {"Authorization": "Bearer client-token"}}

    headers = resolve_server_headers(
        server, user_token=None, client_headers=client_headers
    )

    assert headers is not None
    assert headers == {"Authorization": "Bearer client-token"}


def test_resolve_server_headers_with_kubernetes_placeholder():
    """Test resolve_server_headers replaces kubernetes placeholder with user token."""
    from ols.constants import MCP_KUBERNETES_PLACEHOLDER
    from ols.utils.mcp_utils import resolve_server_headers

    server = MCPServerConfig(
        name="test-server",
        url="http://test:8080/mcp",
        headers={"Authorization": "kubernetes"},
    )
    server._resolved_headers = {"Authorization": MCP_KUBERNETES_PLACEHOLDER}

    headers = resolve_server_headers(
        server, user_token="user-k8s-token", client_headers=None  # noqa: S106 # nosec
    )

    assert headers is not None
    assert headers == {"Authorization": "Bearer user-k8s-token"}


def test_resolve_server_headers_missing_client_headers():
    """Test resolve_server_headers returns None when client headers missing."""
    from ols.constants import MCP_CLIENT_PLACEHOLDER
    from ols.utils.mcp_utils import resolve_server_headers

    server = MCPServerConfig(
        name="test-server",
        url="http://test:8080/mcp",
        headers={"Authorization": "_client_"},
    )
    server._resolved_headers = {"Authorization": MCP_CLIENT_PLACEHOLDER}

    # No client headers provided
    headers = resolve_server_headers(server, user_token=None, client_headers=None)

    assert headers is None


def test_resolve_server_headers_missing_kubernetes_token():
    """Test resolve_server_headers returns None when kubernetes token missing."""
    from ols.constants import MCP_KUBERNETES_PLACEHOLDER
    from ols.utils.mcp_utils import resolve_server_headers

    server = MCPServerConfig(
        name="test-server",
        url="http://test:8080/mcp",
        headers={"Authorization": "kubernetes"},
    )
    server._resolved_headers = {"Authorization": MCP_KUBERNETES_PLACEHOLDER}

    # No user token provided
    headers = resolve_server_headers(server, user_token=None, client_headers=None)

    assert headers is None


def test_resolve_server_headers_with_multiple_client_header_dicts():
    """Test resolve_server_headers handles multiple headers in dict."""
    from ols.constants import MCP_CLIENT_PLACEHOLDER
    from ols.utils.mcp_utils import resolve_server_headers

    server = MCPServerConfig(
        name="test-server",
        url="http://test:8080/mcp",
        headers={"Authorization": "_client_", "X-Custom": "_client_"},
    )
    server._resolved_headers = {
        "Authorization": MCP_CLIENT_PLACEHOLDER,
        "X-Custom": MCP_CLIENT_PLACEHOLDER,
    }

    client_headers = {
        "test-server": {
            "Authorization": "Bearer token",
            "X-Custom": "custom-value",
        }
    }

    headers = resolve_server_headers(
        server, user_token=None, client_headers=client_headers
    )

    assert headers is not None
    assert headers == {"Authorization": "Bearer token", "X-Custom": "custom-value"}


def test_resolve_server_headers_client_does_not_override_static_config():
    """Test client headers don't override static server-configured headers."""
    from ols.utils.mcp_utils import resolve_server_headers

    server = MCPServerConfig(
        name="test-server",
        url="http://test:8080/mcp",
        headers={"Authorization": "Bearer config-token"},
    )
    server._resolved_headers = {"Authorization": "Bearer config-token"}

    # Client provides different authorization (should be ignored for non-placeholder)
    client_headers = {"test-server": {"Authorization": "Bearer client-token"}}

    headers = resolve_server_headers(
        server, user_token=None, client_headers=client_headers
    )

    assert headers is not None
    # Config header should be used (not client)
    assert headers == {"Authorization": "Bearer config-token"}


def test_resolve_server_headers_mixed_placeholders():
    """Test resolve_server_headers with mix of kubernetes and client placeholders."""
    from ols.constants import MCP_CLIENT_PLACEHOLDER, MCP_KUBERNETES_PLACEHOLDER
    from ols.utils.mcp_utils import resolve_server_headers

    server = MCPServerConfig(
        name="test-server",
        url="http://test:8080/mcp",
        headers={"Authorization": "kubernetes", "X-API-Key": "_client_"},
    )
    server._resolved_headers = {
        "Authorization": MCP_KUBERNETES_PLACEHOLDER,
        "X-API-Key": MCP_CLIENT_PLACEHOLDER,
    }

    client_headers = {"test-server": {"X-API-Key": "api-key-123"}}

    headers = resolve_server_headers(
        server,
        user_token="k8s-token",  # noqa: S106 # nosec
        client_headers=client_headers,
    )

    assert headers is not None
    assert headers == {"Authorization": "Bearer k8s-token", "X-API-Key": "api-key-123"}


def test_resolve_server_headers_no_placeholders():
    """Test resolve_server_headers with direct header values (no placeholders)."""
    from ols.utils.mcp_utils import resolve_server_headers

    server = MCPServerConfig(
        name="test-server",
        url="http://test:8080/mcp",
        headers={"Authorization": "Bearer static-token"},
    )
    server._resolved_headers = {"Authorization": "Bearer static-token"}

    headers = resolve_server_headers(server, user_token=None, client_headers=None)

    assert headers is not None
    assert headers == {"Authorization": "Bearer static-token"}
