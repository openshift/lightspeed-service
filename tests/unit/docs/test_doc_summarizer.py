"""Unit tests for DocsSummarizer class."""

from unittest.mock import MagicMock, patch

from ols.app.models.config import ReferenceContent
from ols.src.cache.conversation import Conversation
from ols.src.query_helpers.docs_summarizer import DocsSummarizer, QueryHelper
from ols.utils import config
from tests.mock_classes.langchain_interface import mock_langchain_interface
from tests.mock_classes.llm_chain import mock_llm_chain
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


@patch("ols.src.query_helpers.docs_summarizer.LLMLoader", new=mock_llm_loader(None))
@patch("ols.src.query_helpers.docs_summarizer.ServiceContext.from_defaults")
@patch("ols.src.query_helpers.docs_summarizer.StorageContext.from_defaults")
@patch("ols.src.query_helpers.docs_summarizer.LLMChain", new=mock_llm_chain(None))
def test_chat_engine_with_history(storage_context, service_context):
    """Basic test with history for lanchain chat engine using mocked index and query engine."""
    config.init_empty_config()
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.product_docs_index_path = "./invalid_dir"
    config.ols_config.reference_content.product_docs_index_id = "product"
    summarizer = DocsSummarizer()
    question = "What's the ultimate question with answer 42?"
    history = [
        Conversation(user_message="user hello", assistant_message="assistant hello")
    ]
    _, chat_history = summarizer.get_chat_engine_langchain("1234", question, history)

    assert len(chat_history) == 2
    assert chat_history[0].role == "user"
    assert chat_history[0].content == "user hello"
    assert chat_history[1].role == "assistant"
    assert chat_history[1].content == "assistant hello"


@patch("ols.src.query_helpers.docs_summarizer.LLMLoader", new=mock_llm_loader(None))
@patch("ols.src.query_helpers.docs_summarizer.ServiceContext.from_defaults")
@patch("ols.src.query_helpers.docs_summarizer.StorageContext.from_defaults")
@patch("ols.src.query_helpers.docs_summarizer.LLMChain", new=mock_llm_chain(None))
def test_chat_engine_no_history(storage_context, service_context):
    """Basic test with history for lanchain chat engine using mocked index and query engine."""
    config.init_empty_config()
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.product_docs_index_path = "./invalid_dir"
    config.ols_config.reference_content.product_docs_index_id = "product"
    summarizer = DocsSummarizer()
    question = "What's the ultimate question with answer 42?"
    history = []
    _, chat_history = summarizer.get_chat_engine_langchain("1234", question, history)

    assert len(chat_history) == 0


@patch("ols.src.query_helpers.docs_summarizer.LLMLoader", new=mock_llm_loader(None))
@patch("ols.src.query_helpers.docs_summarizer.ServiceContext.from_defaults")
@patch("ols.src.query_helpers.docs_summarizer.StorageContext.from_defaults")
@patch("ols.src.query_helpers.docs_summarizer.LLMChain", new=mock_llm_chain(None))
def test_get_docs_no_rag_with_history(storage_context, service_context):
    """Basic test with history for lanchain chat engine using mocked index and query engine."""
    config.init_empty_config()
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.product_docs_index_path = "./invalid_dir"
    config.ols_config.reference_content.product_docs_index_id = "product"
    summarizer = DocsSummarizer()
    question = "What's the ultimate question with answer 42?"
    history = [
        Conversation(user_message="user hello", assistant_message="assistant hello")
    ]
    text, doc = summarizer.get_docs_no_rag("1234", question, history)

    assert "user hello" in str(text)
    assert "assistant hello" in str(text)
    assert doc == ""


@patch("ols.src.query_helpers.docs_summarizer.LLMLoader", new=mock_llm_loader(None))
@patch("ols.src.query_helpers.docs_summarizer.ServiceContext.from_defaults")
@patch("ols.src.query_helpers.docs_summarizer.StorageContext.from_defaults")
@patch("ols.src.query_helpers.docs_summarizer.LLMChain", new=mock_llm_chain(None))
def test_get_docs_no_rag_empty_history(storage_context, service_context):
    """Basic test with history for lanchain chat engine using mocked index and query engine."""
    config.init_empty_config()
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.product_docs_index_path = "./invalid_dir"
    config.ols_config.reference_content.product_docs_index_id = "product"
    summarizer = DocsSummarizer()
    question = "What's the ultimate question with answer 42?"
    history = []
    text, doc = summarizer.get_docs_no_rag("1234", question, history)

    assert "user" not in str(text)
    assert "assistant" not in str(text)
    assert doc == ""


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
    ml = mock_llm_chain({"text": "ai answer"})
    ch = [Conversation(user_message="test", assistant_message="test")]
    ml.invoke = MagicMock(return_value={"text": "success"})
    with patch("ols.src.query_helpers.docs_summarizer.LLMChain", new=ml):
        summary, documents = summarizer.summarize("1234", question, ch)
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
    ml = mock_llm_chain({"text": "ai answer"})
    ml.invoke = MagicMock(return_value={"text": "success"})
    with patch("ols.src.query_helpers.docs_summarizer.LLMChain", new=ml):
        summary, documents = summarizer.summarize("1234", question)
    assert (
        "The following response was generated without access to reference content"
        in str(summary)
    )
    assert len(documents) == 0
