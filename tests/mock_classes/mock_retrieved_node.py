"""Class to create mock data for retrieved node."""

from typing import Any, Optional


class MockRetrievedNode:
    """Class to create mock data for retrieved node."""

    score: Optional[float] = None

    def __init__(self, node_detail: dict):
        """Initialize the class instance."""
        super()
        self._text = node_detail["text"]
        self.score = node_detail["score"]
        self._metadata = node_detail["metadata"]

    def get_text(self) -> str:
        """Mock get_text."""
        return self._text

    def get_score(self, raise_error: bool = False) -> float:
        """Mock method to retrieve score."""
        if self.score is None:
            if raise_error:
                raise ValueError("Score not set.")
            return 0.0
        return self.score

    @property
    def metadata(self) -> dict[str, Any]:
        """Mock metadata."""
        return self._metadata
