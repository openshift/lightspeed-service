"""Reranker for post-processing the Vector DB search results."""

import logging

from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)


def rerank(retrieved_nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """Rerank Vector DB search results."""
    message = f"reranker.rerank() is called with {len(retrieved_nodes)} result(s)."
    logger.debug(message)
    return retrieved_nodes
