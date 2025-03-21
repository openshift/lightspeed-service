"""Mocked ComposableGraph."""

from typing import Any, Optional, Sequence, Type

from llama_index.core import StorageContext
from llama_index.core.indices.base import BaseIndex

from tests.mock_classes.mock_llama_index import MockLlamaIndex


class MockComposableGraph:
    """Mocked ComposableGraph.

    Example usage in a test:

        @patch(
            "llama_index.core.indices.composability.graph.ComposableGraph.from_indices",
            new=MockComposableGraph,
        )
        def test_xyz():

        or within test function or test method:
        with patch(
            "llama_index.core.indices.composability.graph.ComposableGraph.from_indices",
            new=MockComposableGraph,
        ):
            some test steps
    """

    def __init__(self, *args, **kwargs):
        """Store all provided arguments for later usage."""
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def from_indices(
        cls,
        root_index_cls: Type[BaseIndex],
        children_indices: Sequence[BaseIndex],
        index_summaries: Optional[Sequence[str]] = None,
        storage_context: Optional[StorageContext] = None,
        **kwargs: Any,
    ) -> "MockComposableGraph":
        """Create a new instance of MockComposableGraph."""
        return MockComposableGraph(
            root_index_cls=root_index_cls,
            children_indices=children_indices,
            index_summaries=index_summaries,
            storage_context=storage_context,
            **kwargs,
        )

    def get_index(self, index_struct_id: Optional[str] = None) -> MockLlamaIndex:
        """Return a MockLlamaIndex instead of BaseIndex."""
        return MockLlamaIndex()
