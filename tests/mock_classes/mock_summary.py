"""Mocked summary returned from query engine."""


class MockSummary:
    """Mocked summary returned from query engine."""

    def __init__(self, query, nodes=None):
        """Initialize all required object attributes."""
        if nodes is None:
            nodes = []

        self.query = query
        self.source_nodes = nodes

    def __str__(self):
        """Return string representation that is used by DocsSummarizer."""
        return f"Summary for query '{self.query}'"
