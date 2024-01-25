"""Mocked (llama) index."""

from .mock_query_engine import MockQueryEngine


class MockLlamaIndex:
    """Mocked (llama) index.

    Example usage in a test:

        @patch("ols.src.query_helpers.docs_summarizer.load_index_from_storage", new=MockLlamaIndex)
        def test_xyz():

        or within test function or test method:
        with patch("ols.src.query_helpers.docs_summarizer.load_index_from_storage", new=MockLlamaIndex):
            some test steps
    """  # noqa E501

    def __init__(self, *args, **kwargs):
        """Store all provided arguments for later usage."""
        self.args = args
        self.kwargs = kwargs

    def as_query_engine(self, **kwargs):
        """Return mocked query engine."""
        return MockQueryEngine(kwargs)
