"""Class to create mock data for retrieved node."""

from typing import Any


class MockRetrievedNode:
    """Class to create mock data for retrieved node."""

    def __init__(self, node_detail: dict):
        """Initialize the class instance."""
        self._text = node_detail["text"]
        self._score = node_detail["score"]
        self._metadata = node_detail["metadata"]

    def get_text(self) -> str:
        """Mock get_text."""
        return self._text

    @property
    def score(self) -> float:
        """Mock score."""
        return self._score

    def get_score(self, **kwargs) -> float:
        """Mock method to retrieve score."""
        return self.score

    @property
    def metadata(self) -> dict[str, Any]:
        """Mock metadata."""
        return self._metadata
