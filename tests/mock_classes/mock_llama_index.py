"""Mocked (llama) index."""

from .mock_query_engine import MockQueryEngine


class MockLlamaIndex:
    """Mocked (llama) index."""

    def __init__(self, *args, **kwargs):
        """Store all provided arguments for later usage."""
        self.args = args
        self.kwargs = kwargs

    def as_query_engine(self, **kwargs):
        """Return mocked query engine."""
        return MockQueryEngine(kwargs)
