"""Unit tests for DocsSummarizer class."""

from unittest.mock import patch

from ols import constants
from ols.src.query_helpers.docs_summarizer import DocsSummarizer, QueryHelper
from ols.utils import config, suid
from tests.mock_classes.langchain_interface import mock_langchain_interface
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader
from tests.mock_classes.mock_llama_index import MockLlamaIndex

conversation_id = suid.get_suid()


def test_is_query_helper_subclass():
    """Test that DocsSummarizer is a subclass of QueryHelper."""
    assert issubclass(DocsSummarizer, QueryHelper)


@patch("ols.src.query_helpers.docs_summarizer.LLMChain", new=mock_llm_chain(None))
def test_summarize():
    """Basic test for DocsSummarizer using mocked index and query engine."""
    config.init_config("tests/config/valid_config.yaml")
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    question = "What's the ultimate question with answer 42?"
    rag_index = MockLlamaIndex()
    history = []  # empty history
    summary = summarizer.summarize(conversation_id, question, rag_index, history)
    assert question in summary["response"]
    documents = summary["referenced_documents"]
    assert len(documents) > 0
    assert f"{constants.OCP_DOCS_ROOT_URL}/docs/test.html" in documents
    assert not summary["history_truncated"]


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
