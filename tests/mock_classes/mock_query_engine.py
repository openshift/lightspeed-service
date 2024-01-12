"""Mocked query engine."""

from .mock_summary import MockSummary


class MockQueryEngine:
    """Mocked query engine."""

    def __init__(self, *args, **kwargs):
        """Constructor accepting all parameters."""
        self.args = args
        self.kwargs = kwargs

    def query(self, query):
        """Returns summary for given query."""
        return MockSummary(query)
