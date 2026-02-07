"""Unit tests for DocsSummarizer class."""

import json
import logging
import re
from math import ceil
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.ai import AIMessageChunk

from ols import config
from ols.app.models.config import MCPServerConfig, MCPServers
from ols.constants import TOKEN_BUFFER_WEIGHT
from ols.utils.token_handler import TokenHandler
from tests.mock_classes.mock_tools import mock_tools_map

# needs to be setup there before is_user_authorized is imported
config.ols_config.authentication_config.module = "k8s"

from ols.app.models.config import LoggingConfig  # noqa: E402
from ols.app.models.models import CacheEntry  # noqa: E402
from ols.src.query_helpers.docs_summarizer import (  # noqa: E402
    DocsSummarizer,
    QueryHelper,
)
from ols.utils import suid  # noqa: E402
from ols.utils.logging_configurator import configure_logging  # noqa: E402
from tests import constants  # noqa: E402
from tests.mock_classes.mock_langchain_interface import (  # noqa: E402
    mock_langchain_interface,
)
from tests.mock_classes.mock_llm_loader import mock_llm_loader  # noqa: E402
from tests.mock_classes.mock_retrievers import MockRetriever  # noqa: E402

CONVERSATION_ID = suid.get_suid()


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
        patch("ols.config.conversation_cache.get") as mock_cache_get,
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        rag_retriever = MockRetriever()

        # first call with history in cache
        mock_cache_get.return_value = (
            [CacheEntry(query=HumanMessage("What is Kubernetes?"))],
            False,
        )
        with patch(
            "ols.src.query_helpers.docs_summarizer.TokenHandler.limit_conversation_history",
            return_value=([], False),
        ) as token_handler:
            summary1 = summarizer.create_response(
                question, rag_retriever, "user-id", "conv-id"
            )
            # History is fetched internally and truncated
            token_handler.assert_called_once()
            check_summary_result(summary1, question)

        # second call without history in cache
        mock_cache_get.return_value = ([], False)
        with patch(
            "ols.src.query_helpers.docs_summarizer.TokenHandler.limit_conversation_history",
            return_value=([], False),
        ) as token_handler:
            summary2 = summarizer.create_response(
                question, rag_retriever, "user-id", "conv-id2"
            )
            token_handler.assert_called_once()
            check_summary_result(summary2, question)


def test_summarize_truncation():
    """Basic test for DocsSummarizer to check if truncation is done."""
    with (
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.config.conversation_cache.get") as mock_cache_get,
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        rag_retriever = MockRetriever()

        # too long history - mock cache returning huge history
        # Create many CacheEntry objects to trigger truncation
        history = [
            CacheEntry(
                query=HumanMessage("What is Kubernetes?" * 100),
                response=AIMessage(
                    "Kubernetes is a container orchestration system." * 100
                ),
            )
        ] * 100  # 100 large entries should definitely trigger truncation
        mock_cache_get.return_value = (history, False)

        summary = summarizer.create_response(
            question, rag_retriever, "user-id", "conv-id"
        )

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
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._build_mcp_config",
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
        patch(
            "ols.src.query_helpers.docs_summarizer.MultiServerMCPClient"
        ) as mock_mcp_client_cls,
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm"
        ) as mock_invoke,
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._build_mcp_config",
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


def test_summarize_entries_success(_setup):
    """Test _summarize_entries with successful LLM summarization."""
    entries = [
        CacheEntry(
            query=HumanMessage(content="What is Kubernetes?"),
            response=AIMessage(
                content="Kubernetes is a container orchestration platform."
            ),
        ),
        CacheEntry(
            query=HumanMessage(content="How do I create a pod?"),
            response=AIMessage(content="Use kubectl create pod command."),
        ),
    ]

    mock_llm_response = AIMessage(
        content="Summary: Discussion about Kubernetes basics and pod creation."
    )

    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    summarizer.bare_llm = MagicMock()
    summarizer.bare_llm.invoke.return_value = mock_llm_response

    summary = summarizer._summarize_entries(entries)

    assert summary == "Summary: Discussion about Kubernetes basics and pod creation."
    summarizer.bare_llm.invoke.assert_called_once()


def test_summarize_entries_empty(_setup):
    """Test _summarize_entries with empty entries list."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

    summary = summarizer._summarize_entries([])

    assert summary is None


def test_summarize_entries_llm_failure(_setup, caplog):
    """Test _summarize_entries when LLM fails."""
    entries = [
        CacheEntry(
            query=HumanMessage(content="Test query"),
            response=AIMessage(content="Test response"),
        ),
    ]

    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    summarizer.bare_llm = MagicMock()
    summarizer.bare_llm.invoke.side_effect = Exception("LLM error")

    summary = summarizer._summarize_entries(entries)

    assert summary is None
    assert "Failed to summarize conversation entries" in caplog.text


def test_compress_conversation_history_no_compression_needed(_setup):
    """Test _compress_conversation_history with 5 or fewer entries."""
    conversation_id = suid.get_suid()
    user_id = "test_user"

    cache_entries = [
        CacheEntry(
            query=HumanMessage(content=f"Query {i}"),
            response=AIMessage(content=f"Response {i}"),
        )
        for i in range(5)
    ]

    with (
        patch("ols.config.conversation_cache.get") as mock_cache_get,
        patch("ols.config.conversation_cache.delete") as mock_cache_delete,
        patch("ols.config.conversation_cache.insert_or_append") as mock_cache_insert,
    ):
        mock_cache_get.return_value = (cache_entries, False)

        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

        result = summarizer._compress_conversation_history(
            user_id, conversation_id, skip_user_id_check=True
        )

        assert result == cache_entries
        # Cache should not be modified
        mock_cache_delete.assert_not_called()
        mock_cache_insert.assert_not_called()


def test_compress_conversation_history_successful_compression(_setup):
    """Test _compress_conversation_history with successful compression."""
    conversation_id = suid.get_suid()
    user_id = "test_user"

    cache_entries = [
        CacheEntry(
            query=HumanMessage(content=f"Query {i}"),
            response=AIMessage(content=f"Response {i}"),
        )
        for i in range(10)
    ]

    summary_text = "Summary of first 5 conversations"

    with (
        patch("ols.config.conversation_cache.get") as mock_cache_get,
        patch("ols.config.conversation_cache.delete") as mock_cache_delete,
        patch("ols.config.conversation_cache.insert_or_append") as mock_cache_insert,
    ):
        mock_cache_get.return_value = (cache_entries, False)

        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

        with patch.object(summarizer, "_summarize_entries", return_value=summary_text):
            result = summarizer._compress_conversation_history(
                user_id, conversation_id, skip_user_id_check=True
            )

            assert len(result) == 6
            assert result[0].query.content == "[Previous conversation summary]"
            assert result[0].response.content == summary_text

            mock_cache_delete.assert_called_once_with(user_id, conversation_id, True)
            # Should be called 6 times (1 summary + 5 entries)
            assert mock_cache_insert.call_count == 6


def test_compress_conversation_history_summarization_failure(_setup, caplog):
    """Test _compress_conversation_history when summarization fails."""
    conversation_id = suid.get_suid()
    user_id = "test_user"

    cache_entries = [
        CacheEntry(
            query=HumanMessage(content=f"Query {i}"),
            response=AIMessage(content=f"Response {i}"),
        )
        for i in range(10)
    ]

    with (
        patch("ols.config.conversation_cache.get") as mock_cache_get,
        patch("ols.config.conversation_cache.delete") as mock_cache_delete,
        patch("ols.config.conversation_cache.insert_or_append") as mock_cache_insert,
    ):
        mock_cache_get.return_value = (cache_entries, False)

        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

        with patch.object(summarizer, "_summarize_entries", return_value=None):
            result = summarizer._compress_conversation_history(
                user_id, conversation_id, skip_user_id_check=True
            )

            assert len(result) == 5
            assert result == cache_entries[-5:]

            mock_cache_delete.assert_not_called()
            mock_cache_insert.assert_not_called()
            assert "Summarization failed" in caplog.text


def test_compress_conversation_history_cache_update_failure(_setup, caplog):
    """Test _compress_conversation_history when cache update fails."""
    conversation_id = suid.get_suid()
    user_id = "test_user"

    cache_entries = [
        CacheEntry(
            query=HumanMessage(content=f"Query {i}"),
            response=AIMessage(content=f"Response {i}"),
        )
        for i in range(10)
    ]

    summary_text = "Summary of conversations"

    with (
        patch("ols.config.conversation_cache.get") as mock_cache_get,
        patch("ols.config.conversation_cache.delete") as mock_cache_delete,
    ):
        mock_cache_get.return_value = (cache_entries, False)
        mock_cache_delete.side_effect = Exception("Cache error")

        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

        with patch.object(summarizer, "_summarize_entries", return_value=summary_text):
            result = summarizer._compress_conversation_history(
                user_id, conversation_id, skip_user_id_check=True
            )

            assert len(result) == 5
            assert result == cache_entries[-5:]
            assert "Failed to update cache with compressed history" in caplog.text


def test_summarize_entries_with_retry_on_transient_error(_setup, caplog):
    """Test _summarize_entries retries on transient errors."""
    entries = [
        CacheEntry(
            query=HumanMessage(content="What is Kubernetes?"),
            response=AIMessage(
                content="Kubernetes is a container orchestration platform."
            ),
        ),
    ]

    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    summarizer.bare_llm = MagicMock()

    # First call fails with timeout, second succeeds
    mock_response = AIMessage(content="Summary of Kubernetes discussion")
    summarizer.bare_llm.invoke.side_effect = [
        Exception("Connection timeout"),
        mock_response,
    ]

    with patch("ols.src.query_helpers.docs_summarizer.time.sleep"):
        summary = summarizer._summarize_entries(entries)

    assert summary == "Summary of Kubernetes discussion"
    assert summarizer.bare_llm.invoke.call_count == 2
    assert "Transient error on attempt 1/3" in caplog.text
    assert "Connection timeout" in caplog.text


def test_summarize_entries_with_retry_exhausted(_setup, caplog):
    """Test _summarize_entries gives up after max retries on transient errors."""
    entries = [
        CacheEntry(
            query=HumanMessage(content="Test query"),
            response=AIMessage(content="Test response"),
        ),
    ]

    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    summarizer.bare_llm = MagicMock()

    # All attempts fail with timeout
    summarizer.bare_llm.invoke.side_effect = Exception("Rate limit exceeded")

    with patch("ols.src.query_helpers.docs_summarizer.time.sleep"):
        summary = summarizer._summarize_entries(entries)

    assert summary is None
    assert summarizer.bare_llm.invoke.call_count == 3
    assert "Transient error on attempt 1/3" in caplog.text
    assert "Transient error on attempt 2/3" in caplog.text
    assert "Failed after 3 attempt(s)" in caplog.text


def test_summarize_entries_no_retry_on_permanent_error(_setup, caplog):
    """Test _summarize_entries does not retry on permanent errors."""
    entries = [
        CacheEntry(
            query=HumanMessage(content="Test query"),
            response=AIMessage(content="Test response"),
        ),
    ]

    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    summarizer.bare_llm = MagicMock()

    # Permanent error (authentication failure)
    summarizer.bare_llm.invoke.side_effect = Exception("Authentication failed")

    summary = summarizer._summarize_entries(entries)

    assert summary is None
    # Should only try once for non-transient errors
    assert summarizer.bare_llm.invoke.call_count == 1
    assert "Failed after 1 attempt(s)" in caplog.text
    assert "Transient error" not in caplog.text


def test_build_mcp_config_transport_is_streamable_http():
    """Test _build_mcp_config sets transport to streamable_http for all servers."""
    server1 = MCPServerConfig(name="server1", url="http://server1:8080/mcp")
    server1._resolved_headers = {}

    server2 = MCPServerConfig(name="server2", url="http://server2:9090/mcp", timeout=30)
    server2._resolved_headers = {}

    mock_mcp_servers = MCPServers(servers=[server1, server2])

    with patch("ols.src.query_helpers.docs_summarizer.config") as mock_config:
        mock_config.mcp_servers = mock_mcp_servers

        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        mcp_config = summarizer._build_mcp_config()

        assert "server1" in mcp_config
        assert "server2" in mcp_config

        assert mcp_config["server1"]["transport"] == "streamable_http"
        assert mcp_config["server1"]["url"] == "http://server1:8080/mcp"

        assert mcp_config["server2"]["transport"] == "streamable_http"
        assert mcp_config["server2"]["url"] == "http://server2:9090/mcp"
        assert mcp_config["server2"]["timeout"] == 30
