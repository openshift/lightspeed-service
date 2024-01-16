"""Mocked query engine."""

from .mock_summary import MockSummary


class MockQueryEngine:
    """Mocked query engine."""

    def __init__(self, *args, **kwargs):
        """Store all provided arguments for later usage."""
        self.args = args
        self.kwargs = kwargs

    def query(self, query):
        """Return summary for given query."""
        return MockSummary(query)
