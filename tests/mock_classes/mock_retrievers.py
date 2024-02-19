"""Mocks for VectorStore and VectorStoreRetriever."""

from langchain_core.documents import Document

from tests.unit.utils.test_token_handler import MockRetrievedNode


class MockRetriever:
    """Mocked retriever."""

    def __init__(self, *args, **kwargs):
        """Store all provided arguments for later usage."""
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def retrieve(*args):
        """Return summary for given query."""
        # fill-in some documents for more realistic tests
        return [
            MockRetrievedNode(
                {
                    "text": "a text text text text",
                    "score": 0.6,
                    "metadata": {"file_path": "/docs/test.txt"},
                }
            )
        ]


class MockVectorStore:
    """Mock for VectorStore."""

    @staticmethod
    def as_retriever():
        """Pass."""


class MockVectorRetriever:
    """Mock for VectorStoreRetriever."""

    @staticmethod
    def get_relevant_documents(*args, **kwargs):
        """Return the docs."""
        return [
            Document(page_content="foo", metadata={"page": 1, "source": "adhoc"}),
        ]


def mock_retriever(*args, **kwargs):
    """Return the mock retriever."""
    return MockVectorRetriever()


def mock_null_value_retriever(*args, **kwargs):
    """Return no VectorStoreRetriever."""
    return
