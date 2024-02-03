"""Mocked chat engine."""

from .mock_response import MockResponse


class MockChatEngine:
    """Mocked chat engine."""

    def __init__(self, *args, **kwargs):
        """Store all provided arguments for later usage."""
        self.args = args
        self.kwargs = kwargs

    def chat(self, query):
        """Return summary for given query."""
        return MockResponse(query)
