"""Unit tests for DocsSummarizer class."""

from unittest.mock import patch

from ols.app.models.config import ReferenceContent
from ols.src.query_helpers.docs_summarizer import DocsSummarizer, QueryHelper
from ols.utils import config
from tests.mock_classes.langchain_interface import mock_langchain_interface
from tests.mock_classes.llm_loader import mock_llm_loader
from tests.mock_classes.mock_llama_index import MockLlamaIndex


def test_is_query_helper_subclass():
    """Test that DocsSummarizer is a subclass of QueryHelper."""
    assert issubclass(DocsSummarizer, QueryHelper)


@patch("ols.src.query_helpers.docs_summarizer.LLMLoader", new=mock_llm_loader(None))
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
    summarizer = DocsSummarizer()
    question = "What's the ultimate question with answer 42?"
    history = None
    summary, documents = summarizer.summarize("1234", question, history)
    assert question in str(summary)
    assert len(documents) == 0


@patch(
    "ols.src.query_helpers.docs_summarizer.LLMLoader",
    new=mock_llm_loader(mock_langchain_interface("test response")()),
)
@patch("ols.src.query_helpers.docs_summarizer.ServiceContext.from_defaults")
@patch("ols.src.query_helpers.docs_summarizer.StorageContext.from_defaults")
@patch(
    "ols.src.query_helpers.docs_summarizer.load_index_from_storage", new=MockLlamaIndex
)
def test_summarize_no_reference_content(storage_context, service_context):
    """Basic test for DocsSummarizer using mocked index and query engine."""
    config.init_empty_config()
    config.ols_config.reference_content = ReferenceContent(None)
    summarizer = DocsSummarizer()
    question = "What's the ultimate question with answer 42?"
    summary, documents = summarizer.summarize("1234", question)
    assert "success" in str(summary)
    assert len(documents) == 0


@patch(
    "ols.src.query_helpers.docs_summarizer.LLMLoader",
    new=mock_llm_loader(mock_langchain_interface("test response")()),
)
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
    summarizer = DocsSummarizer()
    question = "What's the ultimate question with answer 42?"
    conversation_id = "01234567-89ab-cdef-0123-456789abcdef"
    summary, documents = summarizer.summarize(conversation_id, question)
    assert (
        "The following response was generated without access to reference content"
        in str(summary)
    )
    assert len(documents) == 0
