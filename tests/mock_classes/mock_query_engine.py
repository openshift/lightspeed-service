"""Mocked query engine."""

from .mock_summary import MockSummary


class Node:
    """Node containing source node metadata."""

    def __init__(self, file_path):
        """Initialize file_path metadata."""
        self.metadata = {"file_path": file_path}


class SourceNode:
    """Node containing one reference to document."""

    def __init__(self, file_path):
        """Initialize sub-node with metadata."""
        self.node = Node(file_path)


class MockQueryEngine:
    """Mocked query engine."""

    def __init__(self, *args, **kwargs):
        """Store all provided arguments for later usage."""
        self.args = args
        self.kwargs = kwargs

    def query(self, query):
        """Return summary for given query."""
        # fill-in some documents for more realistic tests
        nodes = [
            SourceNode("/docs/test."),
            SourceNode("/known-bugs."),
            SourceNode("/errata."),
        ]
        return MockSummary(query, nodes)
