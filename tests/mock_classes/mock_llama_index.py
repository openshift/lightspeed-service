"""Mocked (llama) index."""

from .mock_query_engine import MockQueryEngine


class MockLlamaIndex:
    """Mocked (llama) index."""

    def __init__(self, *args, **kwargs):
        """Constructor accepting all parameters."""
        self.args = args
        self.kwargs = kwargs

    def as_query_engine(self, **kwargs):
        """Returns mocked query engine."""
        return MockQueryEngine(kwargs)
