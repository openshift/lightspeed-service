"""Hybrid Tools RAG implementation."""

import json
from functools import cached_property
from typing import Any

from langchain_core.tools.structured import StructuredTool
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class ChromaStore:
    """Wrapper for ChromaDB vector database operations."""

    def __init__(self) -> None:
        """Initialize in-memory ChromaDB client and collection."""
        import chromadb  # pylint: disable=import-outside-toplevel

        self.client = chromadb.Client()
        self.coll = self.client.get_or_create_collection(
            "tools", metadata={"hnsw:space": "cosine"}
        )

    def upsert(
        self,
        ids: list[str],
        docs: list[str],
        vectors: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add or update documents with embeddings in the collection.

        Args:
            ids: List of unique identifiers for documents
            docs: List of document texts
            vectors: List of embedding vectors
            metadatas: Optional list of metadata dictionaries (e.g., full tool data)
        """
        self.coll.upsert(
            ids=ids, documents=docs, embeddings=vectors, metadatas=metadatas
        )

    def search_with_scores(
        self,
        vector: list[float],
        k: int,
        allowed_servers: set[str] | None = None,
    ) -> tuple[list[str], list[float], list[dict]]:
        """Search and return IDs, similarity scores, and metadata.

        Args:
            vector: Query embedding vector
            k: Number of results to return
            allowed_servers: Optional set of server names to filter by

        Returns:
            Tuple of (document IDs, similarity scores, metadatas).

            Scores are 0-1 (1=most similar).
        """
        # Build where clause for server filtering
        where = None
        if allowed_servers is not None and allowed_servers:
            where = {"server": {"$in": list(allowed_servers)}}

        results = self.coll.query(
            query_embeddings=[vector],
            n_results=k,
            where=where,
            include=["metadatas", "distances"],
        )
        ids = results["ids"][0]  # type: ignore[index]
        distances = results["distances"][0]  # type: ignore[index]
        metadatas = results["metadatas"][0]  # type: ignore[index]
        # Convert cosine distance to similarity (1 - distance)
        # ChromaDB returns distances, lower is better, so we invert
        similarities = [1 - d for d in distances]
        return ids, similarities, metadatas

    def delete(self, ids: list[str]) -> None:
        """Delete documents from the collection.

        Args:
            ids: List of document IDs to delete
        """
        self.coll.delete(ids=ids)

    def get_all(self) -> dict:
        """Get all documents with their metadata.

        Returns:
            Dictionary with 'ids', 'documents', 'embeddings', and 'metadatas' keys
        """
        return self.coll.get()


class ToolsRAG:
    """Hybrid RAG system for tool retrieval using dense and sparse methods."""

    def __init__(
        self,
        embed_model: str | None = None,
        alpha: float = 0.8,
        top_k: int = 10,
        threshold: float = 0.01,
    ) -> None:
        """Initialize the ToolsRAG system with configuration.

        Args:
            embed_model: Sentence transformer model for embeddings
            alpha: Weight for dense vs sparse retrieval (1.0 = full dense, 0.0 = full sparse)
            top_k: Number of tools to retrieve
            threshold: Minimum similarity threshold for filtering results
        """
        self.embed_model = embed_model or "sentence-transformers/all-mpnet-base-v2"
        self.alpha = alpha
        self.top_k = top_k
        self.threshold = threshold
        self.embedding_model = SentenceTransformer(self.embed_model)
        self.bm25 = None
        self.default_allowed_servers: set[str] = set()  # K8s/static servers

    @cached_property
    def store(self) -> ChromaStore:
        """Lazy initialization of ChromaStore to avoid SQLite dependency at import time."""
        return ChromaStore()

    def set_default_servers(self, servers: list[str]) -> None:
        """Set the default k8s/static servers that are always included.

        Args:
            servers: List of server names to use as defaults
        """
        self.default_allowed_servers = set(servers)

    # Public/functional methods

    def populate_tools(self, tools_list: list[StructuredTool]) -> None:
        """Populate the RAG system with tools.

        ChromaDB automatically handles upsert - if a tool with the same name exists,
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
            # Convert to dict format
            tool_dict = self._convert_langchain_tool_to_dict(tool)

            # Build text representation
            text = self._build_text(tool_dict)

            # Create embedding
            vector = self.embedding_model.encode(text).tolist()

            # Collect all data
            ids.append(f"{tool_dict.get('server', '')}::{tool_dict['name']}")
            dense_docs.append(text)
            vectors.append(vector)
            # Store tool_json AND server at top level for ChromaDB filtering
            metadatas.append(
                {
                    "tool_json": json.dumps(tool_dict),
                    "server": tool_dict.get("server", ""),
                }
            )

        self.store.upsert(ids, dense_docs, vectors, metadatas=metadatas)

        # Rebuild BM25 from ChromaDB
        self._rebuild_bm25()

    def remove_tools(self, tool_names: list[str]) -> None:
        """Remove tools by name.

        Args:
            tool_names: List of tool names to remove
        """
        # Delete from ChromaDB
        self.store.delete(tool_names)

        # Rebuild BM25 from ChromaDB
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
        # Use instance defaults if not specified
        k = k if k is not None else self.top_k
        alpha = alpha if alpha is not None else self.alpha
        threshold = threshold if threshold is not None else self.threshold

        # Combine default servers with client servers
        allowed_servers = self.default_allowed_servers
        if client_servers:
            allowed_servers = allowed_servers | set(client_servers)

        # Encode query
        q_vec = self.embedding_model.encode(query).tolist()

        # Dense (semantic) - rank-based scoring with actual similarities and metadata
        dense_ids, dense_similarities, dense_metadatas = self.store.search_with_scores(
            q_vec, k, allowed_servers=allowed_servers
        )
        dense_scores = {t: 1.0 - i / k for i, t in enumerate(dense_ids)}

        # Create lookup dictionaries from dense results
        similarity_lookup = dict(zip(dense_ids, dense_similarities))
        metadata_lookup = {
            tool_id: json.loads(meta["tool_json"])
            for tool_id, meta in zip(dense_ids, dense_metadatas)
        }

        # Sparse (BM25) - normalized to 0-1 range
        sparse_scores = self._retrieve_sparse_scores(
            query, allowed_servers=allowed_servers
        )
        sparse_ids = sorted(sparse_scores, key=sparse_scores.get, reverse=True)[:k]

        # Fusion (alpha * dense + (1-alpha) * sparse)
        fused = {}
        for t in set(dense_ids + sparse_ids):
            d = dense_scores.get(t, 0)
            s = sparse_scores.get(t, 0)
            fused[t] = alpha * d + (1 - alpha) * s
        final = sorted(fused, key=fused.get, reverse=True)[:k]

        # Group tools by server
        server_tools: dict[str, list[dict[str, Any]]] = {}

        for name in final:
            sim = similarity_lookup.get(name, 0.0)
            if sim >= threshold and name in metadata_lookup:
                tool = metadata_lookup[name]
                # Extract and remove server field
                server = tool.pop("server", None)

                if server:
                    if server not in server_tools:
                        server_tools[server] = []
                    server_tools[server].append(tool)

        return server_tools

    # Private/support methods

    def _convert_langchain_tool_to_dict(self, tool: StructuredTool) -> dict[str, Any]:
        """Convert LangChain tool object to dict format for RAG.

        Args:
            tool: LangChain tool object with name, description, and metadata

        Returns:
            Dictionary with 'name', 'desc', 'params', and 'server' fields
        """
        return {
            "name": tool.name,
            "desc": tool.description or "",
            "params": getattr(tool, "args_schema", None),
            "server": (
                tool.metadata.get("mcp_server") if hasattr(tool, "metadata") else None
            ),
        }

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from ChromaDB documents."""
        # Get all documents from ChromaDB
        all_data = self.store.get_all()
        if not all_data["documents"]:
            self.bm25 = None
            return

        # Build BM25 from documents (already in text representation)
        sparse_docs = [doc.split() for doc in all_data["documents"]]
        self.bm25 = BM25Okapi(sparse_docs)

    def _build_text(self, t: dict) -> str:
        """Build text representation: name + desc only (best performance: 99.1% hit rate)."""
        return f"{t['name']} {t['desc']}"

    def _retrieve_sparse_scores(
        self, query: str, allowed_servers: set[str] | None = None
    ) -> dict[str, float]:
        """Retrieve BM25 scores (normalized to 0-1 range).

        Args:
            query: The query string
            allowed_servers: Optional set of server names to filter by

        Returns:
            Dictionary mapping tool names to normalized scores
        """
        if self.bm25 is None:
            return {}

        # Get all tools from ChromaDB to maintain order consistency with BM25
        all_data = self.store.get_all()
        tool_names = all_data["ids"]
        metadatas = all_data["metadatas"]

        sparse_scores_raw = self.bm25.get_scores(query.split())
        max_score = max(sparse_scores_raw) if max(sparse_scores_raw) > 0 else 1

        # Build result dict with optional server filtering
        result = {}
        for i, (name, score) in enumerate(zip(tool_names, sparse_scores_raw)):
            # Filter by server if specified
            if allowed_servers is not None and allowed_servers:
                tool_json = json.loads(metadatas[i]["tool_json"])
                server = tool_json.get("server", "")
                if server not in allowed_servers:
                    continue

            result[name] = score / max_score

        return result
