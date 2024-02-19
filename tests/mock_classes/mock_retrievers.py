"""Mocks for VectorStore and VectorStoreRetriever."""

from langchain_core.documents import Document


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
