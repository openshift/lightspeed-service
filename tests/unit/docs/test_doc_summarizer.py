"""Unit tests for DocsSummarizer class."""

from unittest.mock import patch

from ols import constants
from ols.app.models.config import ReferenceContent
from ols.src.query_helpers.docs_summarizer import DocsSummarizer, QueryHelper
from ols.utils import config, suid
from tests.mock_classes.langchain_interface import mock_langchain_interface
from tests.mock_classes.llm_loader import mock_llm_loader
from tests.mock_classes.mock_llama_index import MockLlamaIndex

conversation_id = suid.get_suid()


def test_is_query_helper_subclass():
    """Test that DocsSummarizer is a subclass of QueryHelper."""
    assert issubclass(DocsSummarizer, QueryHelper)


@patch("ols.src.query_helpers.docs_summarizer.ServiceContext.from_defaults")
@patch("ols.src.query_helpers.docs_summarizer.StorageContext.from_defaults")
@patch(
    "ols.src.query_helpers.docs_summarizer.load_index_from_storage", new=MockLlamaIndex
)
def test_summarize(storage_context, service_context):
    """Basic test for DocsSummarizer using mocked index and query engine."""
    config.init_empty_config()
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.product_docs_index_path = "./invalid_dir"
    config.ols_config.reference_content.product_docs_index_id = "product"
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    question = "What's the ultimate question with answer 42?"
    history = None
    summary, documents, truncated = summarizer.summarize(
        conversation_id, question, history
    )
    assert question in str(summary)
    assert len(documents) > 0
    assert (
        f"{constants.OCP_DOCS_ROOT_URL}{constants.OCP_DOCS_VERSION}/docs/test.html"
        in documents
    )
    assert (
        f"{constants.OCP_DOCS_ROOT_URL}{constants.OCP_DOCS_VERSION}/errata.html"
        in documents
    )
    assert (
        f"{constants.OCP_DOCS_ROOT_URL}{constants.OCP_DOCS_VERSION}/known-bugs.html"
        in documents
    )
    assert not truncated


@patch("ols.src.query_helpers.docs_summarizer.ServiceContext.from_defaults")
@patch("ols.src.query_helpers.docs_summarizer.StorageContext.from_defaults")
@patch(
    "ols.src.query_helpers.docs_summarizer.load_index_from_storage", new=MockLlamaIndex
)
def test_summarize_no_reference_content(storage_context, service_context):
    """Basic test for DocsSummarizer using mocked index and query engine."""
    config.init_empty_config()
    config.ols_config.reference_content = ReferenceContent(None)
    summarizer = DocsSummarizer(
        llm_loader=mock_llm_loader(mock_langchain_interface("test response")())
    )
    question = "What's the ultimate question with answer 42?"
    summary, documents, truncated = summarizer.summarize(conversation_id, question)
    assert "success" in str(summary)
    assert len(documents) == 0
    assert not truncated


@patch("ols.src.query_helpers.docs_summarizer.ServiceContext.from_defaults")
@patch(
    "ols.src.query_helpers.docs_summarizer.load_index_from_storage", new=MockLlamaIndex
)
def test_summarize_incorrect_directory(service_context):
    """Basic test for DocsSummarizer using mocked index and query engine."""
    config.init_empty_config()
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.product_docs_index_path = "./invalid_dir"
    config.ols_config.reference_content.product_docs_index_id = "product"
    summarizer = DocsSummarizer(
        llm_loader=mock_llm_loader(mock_langchain_interface("test response")())
    )
    question = "What's the ultimate question with answer 42?"
    conversation_id = "01234567-89ab-cdef-0123-456789abcdef"
    summary, documents, truncated = summarizer.summarize(conversation_id, question)
    assert (
        "The following response was generated without access to reference content"
        in str(summary)
    )
    assert len(documents) == 0
    assert not truncated
