"""Unit tests for DocsSummarizer class."""

from unittest.mock import ANY, patch

import pytest

from ols import config
from ols.src.query_helpers.docs_summarizer import DocsSummarizer, QueryHelper
from ols.utils import suid
from tests import constants
from tests.mock_classes.mock_langchain_interface import mock_langchain_interface
from tests.mock_classes.mock_llama_index import MockLlamaIndex
from tests.mock_classes.mock_llm_chain import mock_llm_chain
from tests.mock_classes.mock_llm_loader import mock_llm_loader

conversation_id = suid.get_suid()


def test_is_query_helper_subclass():
    """Test that DocsSummarizer is a subclass of QueryHelper."""
    assert issubclass(DocsSummarizer, QueryHelper)


def check_summary_result(summary, question):
    """Check result produced by DocsSummarizer.summary method."""
    assert question in summary["response"]
    documents = summary["referenced_documents"]
    assert len(documents) > 0
    assert (
        f"{constants.OCP_DOCS_ROOT_URL}/{constants.OCP_DOCS_VERSION}/docs/test.html"
        in [documents[0].docs_url]
    )
    assert not summary["history_truncated"]
    assert "rag_context" in summary


@pytest.fixture(scope="function", autouse=True)
def _setup():
    """Set up config for tests."""
    config.reload_from_yaml_file("tests/config/valid_config.yaml")


@patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4)
@patch("ols.src.query_helpers.docs_summarizer.LLMChain", new=mock_llm_chain(None))
def test_summarize_empty_history():
    """Basic test for DocsSummarizer using mocked index and query engine."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    question = "What's the ultimate question with answer 42?"
    rag_index = MockLlamaIndex()
    history = []  # empty history
    summary = summarizer.summarize(conversation_id, question, rag_index, history)
    check_summary_result(summary, question)


@patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4)
@patch("ols.src.query_helpers.docs_summarizer.LLMChain", new=mock_llm_chain(None))
def test_summarize_no_history():
    """Basic test for DocsSummarizer using mocked index and query engine, no history is provided."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    question = "What's the ultimate question with answer 42?"
    rag_index = MockLlamaIndex()
    # no history is passed into summarize() method
    summary = summarizer.summarize(conversation_id, question, rag_index)
    check_summary_result(summary, question)


@patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4)
@patch("ols.src.query_helpers.docs_summarizer.LLMChain", new=mock_llm_chain(None))
def test_summarize_history_provided():
    """Basic test for DocsSummarizer using mocked index and query engine, history is provided."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    question = "What's the ultimate question with answer 42?"
    history = ["What is Kubernetes?"]
    rag_index = MockLlamaIndex()

    # first call with history provided
    with patch(
        "ols.src.query_helpers.docs_summarizer.TokenHandler.limit_conversation_history",
        return_value=([], False),
    ) as token_handler:
        summary1 = summarizer.summarize(conversation_id, question, rag_index, history)
        token_handler.assert_called_once_with(history, ANY)
        check_summary_result(summary1, question)

    # second call without history provided
    with patch(
        "ols.src.query_helpers.docs_summarizer.TokenHandler.limit_conversation_history",
        return_value=([], False),
    ) as token_handler:
        summary2 = summarizer.summarize(conversation_id, question, rag_index)
        token_handler.assert_called_once_with([], ANY)
        check_summary_result(summary2, question)


@patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4)
@patch("ols.src.query_helpers.docs_summarizer.LLMChain", new=mock_llm_chain(None))
def test_summarize_truncation():
    """Basic test for DocsSummarizer to check if truncation is done."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    question = "What's the ultimate question with answer 42?"
    rag_index = MockLlamaIndex()

    # too long history
    history = ["What is Kubernetes?"] * 10000
    summary = summarizer.summarize(conversation_id, question, rag_index, history)

    # truncation should be done
    assert summary["history_truncated"]


@patch("ols.src.query_helpers.docs_summarizer.LLMChain", new=mock_llm_chain(None))
def test_summarize_no_reference_content():
    """Basic test for DocsSummarizer using mocked index and query engine."""
    summarizer = DocsSummarizer(
        llm_loader=mock_llm_loader(mock_langchain_interface("test response")())
    )
    question = "What's the ultimate question with answer 42?"
    summary = summarizer.summarize(conversation_id, question)
    assert question in summary["response"]
    documents = summary["referenced_documents"]
    assert len(documents) == 0
    assert not summary["history_truncated"]
