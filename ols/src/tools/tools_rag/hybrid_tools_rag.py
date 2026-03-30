"""Hybrid Tools RAG implementation."""

import json
from collections.abc import Callable
from typing import Any

from langchain_core.tools.structured import StructuredTool

from ols.src.rag.hybrid_rag import HybridRAGBase, QdrantStore

__all__ = ["QdrantStore", "ToolsRAG"]


class ToolsRAG(HybridRAGBase):
    """Hybrid RAG system for tool retrieval using dense and sparse methods."""

    _COLLECTION = "tools"

    def __init__(
        self,
        encode_fn: Callable[[str], list[float]],
        alpha: float = 0.8,
        top_k: int = 10,
        threshold: float = 0.01,
    ) -> None:
        """Initialize the ToolsRAG system with configuration.

        Args:
            encode_fn: Function that encodes text into an embedding vector.
            alpha: Weight for dense vs sparse (1.0 = full dense, 0.0 = full sparse).
            top_k: Number of tools to retrieve.
            threshold: Minimum similarity threshold for filtering results.
        """
        super().__init__(
            collection=self._COLLECTION,
            encode_fn=encode_fn,
            alpha=alpha,
            top_k=top_k,
            threshold=threshold,
        )
        self.default_allowed_servers: set[str] = set()

    def set_default_servers(self, servers: list[str]) -> None:
        """Set the default k8s/static servers that are always included.

        Args:
            servers: List of server names to use as defaults
        """
        self.default_allowed_servers = set(servers)

    # Public/functional methods

    def populate_tools(self, tools_list: list[StructuredTool]) -> None:
        """Populate the RAG system with tools.

        Qdrant automatically handles upsert - if a tool with the same name exists,
        it will be updated. Otherwise, it will be added.

        Args:
            tools_list: List of LangChain tool objects from gather_mcp_tools
        """
        # Process all tools in a single loop
        ids = []
        dense_docs = []
        vectors = []
        metadatas = []

        for tool in tools_list:
            tool_dict = self._convert_langchain_tool_to_dict(tool)
            text = self._build_text(tool_dict)

            ids.append(f"{tool_dict.get('server', '')}::{tool_dict['name']}")
            dense_docs.append(text)
            vectors.append(self._encode(text))
            metadatas.append(
                {
                    "tool_json": json.dumps(tool_dict),
                    "server": tool_dict.get("server", ""),
                }
            )

        self._index_documents(ids, dense_docs, vectors, metadatas=metadatas)

    def remove_tools(self, tool_names: list[str]) -> None:
        """Remove tools by name.

        Args:
            tool_names: List of tool names to remove
        """
        self.store.delete(tool_names)
        self._rebuild_bm25()

    def retrieve_hybrid(
        self,
        query: str,
        client_servers: list[str] | None = None,
        k: int | None = None,
        alpha: float | None = None,
        threshold: float | None = None,
    ) -> dict[str, list[dict]]:
        """Retrieve tools using hybrid (dense + sparse) search with RRF.

        Args:
            query: The search query describing what tools are needed
            client_servers: Optional list of client-specific server names to include
            k: Number of results to return (default from instance)
            alpha: Weight for dense vs sparse (1.0=dense only, 0.0=sparse only)
            threshold: Minimum similarity score (default from instance)

        Returns:
            Dictionary mapping server names to lists of tools (without 'server' field)
        """
        k = k if k is not None else self.top_k
        alpha = alpha if alpha is not None else self.alpha
        threshold = threshold if threshold is not None else self.threshold

        allowed_servers = self.default_allowed_servers
        if client_servers:
            allowed_servers = allowed_servers | set(client_servers)

        q_vec = self._encode(query)

        dense, dense_ids, dense_metas = self._dense_scores(
            q_vec, k, allowed_servers=allowed_servers
        )
        metadata_lookup = {
            tool_id: json.loads(meta["tool_json"])
            for tool_id, meta in zip(dense_ids, dense_metas)
        }

        sparse, sparse_metadata = self._retrieve_sparse_scores(
            query, allowed_servers=allowed_servers
        )
        for tool_id, tool_dict in sparse_metadata.items():
            if tool_id not in metadata_lookup:
                metadata_lookup[tool_id] = tool_dict

        fused = self._fuse_scores(dense, sparse, alpha, k)

        server_tools: dict[str, list[dict[str, Any]]] = {}
        for name, score in fused.items():
            if score < threshold:
                continue
            if name not in metadata_lookup:
                continue
            tool = metadata_lookup[name]
            server = tool.pop("server", None)
            if server:
                server_tools.setdefault(server, []).append(tool)

        return server_tools

    # Private/support methods

    def _convert_langchain_tool_to_dict(self, tool: StructuredTool) -> dict[str, Any]:
        """Convert LangChain tool object to dict format for RAG.

        Args:
            tool: LangChain tool object with name, description, and metadata

        Returns:
            Dictionary with 'name', 'desc', 'params', and 'server' fields
        """
        schema = getattr(tool, "args_schema", None)
        if schema is not None and not isinstance(schema, dict):
            schema = schema.model_json_schema()
        return {
            "name": tool.name,
            "desc": tool.description or "",
            "params": schema,
            "server": (
                tool.metadata.get("mcp_server") if hasattr(tool, "metadata") else None
            ),
        }

    def _build_text(self, t: dict) -> str:
        """Build text representation: name + desc only (best performance: 99.1% hit rate)."""
        return f"{t['name']} {t['desc']}"

    def _retrieve_sparse_scores(
        self, query: str, allowed_servers: set[str] | None = None
    ) -> tuple[dict[str, float], dict[str, dict]]:
        """Retrieve BM25 scores (normalized to 0-1 range) and tool metadata.

        Args:
            query: The query string
            allowed_servers: Optional set of server names to filter by

        Returns:
            Tuple of (scores dict, metadata dict) where scores maps tool names
            to normalized BM25 scores and metadata maps tool names to parsed
            tool dictionaries.
        """
        if self.bm25 is None:
            return {}, {}

        all_data = self.store.get_all()
        tool_names = all_data["ids"]
        metadatas = all_data["metadatas"]

        sparse_scores_raw = self.bm25.get_scores(query.split())
        max_score = max(sparse_scores_raw) if max(sparse_scores_raw) > 0 else 1

        scores: dict[str, float] = {}
        metadata: dict[str, dict] = {}
        for i, (name, score) in enumerate(zip(tool_names, sparse_scores_raw)):
            tool_dict = json.loads(metadatas[i]["tool_json"])
            if allowed_servers is not None and allowed_servers:
                server = tool_dict.get("server", "")
                if server not in allowed_servers:
                    continue

            scores[name] = score / max_score
            metadata[name] = tool_dict

        return scores, metadata
