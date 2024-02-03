"""Mocked summary returned from query engine."""


class MockResponse:
    """Mocked Response returned from chat engine."""

    def __init__(self, response, nodes=[]):
        """Initialize all required object attributes."""
        self.response = response
        self.source_nodes = nodes

    def __str__(self):
        """Return string representation that is used by DocsSummarizer."""
        return f"Response for chat '{self.response}'"
