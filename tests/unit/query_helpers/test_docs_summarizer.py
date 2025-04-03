"""Unit tests for DocsSummarizer class."""

import logging
from unittest.mock import ANY, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ols import config
from ols.app.models.models import TokenCounter
from tests.mock_classes.mock_tools import mock_tools_map

# needs to be setup there before is_user_authorized is imported
config.ols_config.authentication_config.module = "k8s"


from ols.app.models.config import LoggingConfig  # noqa:E402
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
from tests.mock_classes.mock_llama_index import MockLlamaIndex  # noqa:E402
from tests.mock_classes.mock_llm_loader import mock_llm_loader  # noqa:E402

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


@pytest.fixture(scope="function", autouse=True)
def _setup():
    """Set up config for tests."""
    config.reload_from_yaml_file("tests/config/valid_config.yaml")


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
        rag_index = MockLlamaIndex()
        history = []  # empty history
        summary = summarizer.create_response(question, rag_index, history)
        check_summary_result(summary, question)


def test_summarize_no_history():
    """Basic test for DocsSummarizer using mocked index and query engine, no history is provided."""
    with (
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        rag_index = MockLlamaIndex()
        # no history is passed into summarize() method
        summary = summarizer.create_response(question, rag_index)
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
        rag_index = MockLlamaIndex()

        # first call with history provided
        with patch(
            "ols.src.query_helpers.docs_summarizer.TokenHandler.limit_conversation_history",
            return_value=([], False),
        ) as token_handler:
            summary1 = summarizer.create_response(question, rag_index, history)
            token_handler.assert_called_once_with(history, ANY)
            check_summary_result(summary1, question)

        # second call without history provided
        with patch(
            "ols.src.query_helpers.docs_summarizer.TokenHandler.limit_conversation_history",
            return_value=([], False),
        ) as token_handler:
            summary2 = summarizer.create_response(question, rag_index)
            token_handler.assert_called_once_with([], ANY)
            check_summary_result(summary2, question)


def test_summarize_truncation():
    """Basic test for DocsSummarizer to check if truncation is done."""
    with patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        rag_index = MockLlamaIndex()

        # too long history
        history = [HumanMessage("What is Kubernetes?")] * 10000
        summary = summarizer.create_response(question, rag_index, history)

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
        rag_index = MockLlamaIndex()
        # no history is passed into create_response() method
        summary = summarizer.create_response(question, rag_index)
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
        if isinstance(item, str):
            generated_content += item

    assert generated_content == question


def test_tool_calling_one_iteration():
    """Test tool calling - stops after one iteration."""
    config.ols_config.introspection_enabled = True
    question = "How many namespaces are there in my cluster ?"

    with patch(
        "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm"
    ) as mock_invoke:
        mock_invoke.side_effect = [
            (
                AIMessage(content="XYZ", response_metadata={"finish_reason": "stop"}),
                TokenCounter(),
            )
        ]
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer.create_response(question)
        assert mock_invoke.call_count == 1


def test_tool_calling_two_iteration():
    """Test tool calling - stops after two iterations."""
    config.ols_config.introspection_enabled = True
    question = "How many namespaces are there in my cluster ?"

    with patch(
        "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm"
    ) as mock_invoke:
        mock_invoke.side_effect = [
            (
                AIMessage(
                    content="", response_metadata={"finish_reason": "tool_calls"}
                ),
                TokenCounter(),
            ),
            (
                AIMessage(content="XYZ", response_metadata={"finish_reason": "stop"}),
                TokenCounter(),
            ),
        ]
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer.create_response(question)
        assert mock_invoke.call_count == 2


def test_tool_calling_force_stop():
    """Test tool calling - force stop."""
    config.ols_config.introspection_enabled = True
    question = "How many namespaces are there in my cluster ?"

    with (
        patch("ols.src.query_helpers.docs_summarizer.MAX_ITERATIONS", 3),
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm"
        ) as mock_invoke,
    ):
        mock_invoke.side_effect = [
            (
                AIMessage(
                    content="", response_metadata={"finish_reason": "tool_calls"}
                ),
                TokenCounter(),
            )
        ] * 4
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer.create_response(question)
        assert mock_invoke.call_count == 3


def test_tool_calling_tool_execution(caplog):
    """Test tool calling - tool execution."""
    caplog.set_level(10)  # Set debug level
    config.ols_config.introspection_enabled = True

    question = "How many namespaces are there in my cluster ?"

    with (
        patch("ols.src.query_helpers.docs_summarizer.MAX_ITERATIONS", 2),
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._get_available_tools"
        ) as mock_tools,
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm"
        ) as mock_invoke,
    ):
        mock_invoke.return_value = (
            AIMessage(
                content="",
                response_metadata={"finish_reason": "tool_calls"},
                tool_calls=[
                    {"name": "get_namespaces_mock", "args": {}, "id": "call_id1"},
                    {"name": "invalid_function_name", "args": {}, "id": "call_id2"},
                ],
            ),
            TokenCounter(),
        )
        mock_tools.return_value = mock_tools_map

        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer.create_response(question)

        assert "Tool: get_namespaces_mock" in caplog.text
        tool_output = mock_tools_map["get_namespaces_mock"].invoke({})
        assert f"Output: {tool_output}" in caplog.text

        assert "Tool: invalid_function_name" in caplog.text
        assert "Error: Tool 'invalid_function_name' not found." in caplog.text

        assert mock_invoke.call_count == 2
