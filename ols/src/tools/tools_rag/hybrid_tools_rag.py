"""Hybrid Tools RAG implementation."""

import json
import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.tools.structured import StructuredTool
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointIdsList,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi


class QdrantStore:
    """Wrapper for in-memory vector database operations backed by Qdrant."""

    _COLLECTION = "tools"

    def __init__(self) -> None:
        """Initialize in-memory Qdrant client."""
        self.client = QdrantClient(location=":memory:")
        self._collection_ready = False

    def _ensure_collection(self, vector_size: int) -> None:
        """Lazily create the vector collection on first upsert."""
        if self._collection_ready:
            return
        self.client.create_collection(
            self._COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        self._collection_ready = True

    @staticmethod
    def _point_id(string_id: str) -> str:
        """Convert an arbitrary string ID to a deterministic UUID for Qdrant."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))

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
        if not vectors:
            return
        self._ensure_collection(len(vectors[0]))

        points = []
        for i, (str_id, doc, vec) in enumerate(zip(ids, docs, vectors)):
            payload: dict[str, Any] = {"_id": str_id, "_document": doc}
            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])
            points.append(
                PointStruct(id=self._point_id(str_id), vector=vec, payload=payload)
            )
        self.client.upsert(self._COLLECTION, points=points)

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
        if not self._collection_ready:
            return [], [], []

        query_filter = None
        if allowed_servers is not None and allowed_servers:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="server", match=MatchAny(any=list(allowed_servers))
                    )
                ]
            )

        results = self.client.query_points(
            self._COLLECTION,
            query=vector,
            limit=k,
            query_filter=query_filter,
        )

        out_ids: list[str] = []
        scores: list[float] = []
        metas: list[dict] = []
        for point in results.points:
            payload = point.payload or {}
            out_ids.append(payload["_id"])
            scores.append(point.score)
            metas.append({k: v for k, v in payload.items() if not k.startswith("_")})
        return out_ids, scores, metas

    def delete(self, ids: list[str]) -> None:
        """Delete documents from the collection.

        Args:
            ids: List of document IDs to delete
        """
        if not self._collection_ready:
            return

        point_ids = [self._point_id(str_id) for str_id in ids]
        self.client.delete(
            self._COLLECTION,
            points_selector=PointIdsList(points=point_ids),
        )

    def get_all(self) -> dict:
        """Get all documents with their metadata.

        Returns:
            Dictionary with 'ids', 'documents', and 'metadatas' keys.
        """
        if not self._collection_ready:
            return {"ids": [], "documents": [], "metadatas": []}

        records, _ = self.client.scroll(self._COLLECTION, limit=10_000)

        out_ids: list[str] = []
        documents: list[str] = []
        metas: list[dict] = []
        for record in records:
            payload = record.payload or {}
            out_ids.append(payload["_id"])
            documents.append(payload.get("_document", ""))
            metas.append({k: v for k, v in payload.items() if not k.startswith("_")})
        return {"ids": out_ids, "documents": documents, "metadatas": metas}


class ToolsRAG:
    """Hybrid RAG system for tool retrieval using dense and sparse methods."""

    def __init__(
        self,
        encode_fn: Callable[[str], list[float]],
        alpha: float = 0.8,
        top_k: int = 10,
        threshold: float = 0.01,
    ) -> None:
        """Initialize the ToolsRAG system with configuration.

        Args:
            encode_fn: Function that encodes text into an embedding vector
            alpha: Weight for dense vs sparse retrieval (1.0 = full dense, 0.0 = full sparse)
            top_k: Number of tools to retrieve
            threshold: Minimum similarity threshold for filtering results
        """
        self.alpha = alpha
        self.top_k = top_k
        self.threshold = threshold
        self._encode = encode_fn
        self.bm25 = None
        self.default_allowed_servers: set[str] = set()
        self.store = QdrantStore()

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
            # Convert to dict format
            tool_dict = self._convert_langchain_tool_to_dict(tool)

            # Build text representation
            text = self._build_text(tool_dict)

            # Create embedding
            vector = self._encode(text)

            # Collect all data
            ids.append(f"{tool_dict.get('server', '')}::{tool_dict['name']}")
            dense_docs.append(text)
            vectors.append(vector)
            # Store tool_json AND server at top level for Qdrant filtering
            metadatas.append(
                {
                    "tool_json": json.dumps(tool_dict),
                    "server": tool_dict.get("server", ""),
                }
            )

        self.store.upsert(ids, dense_docs, vectors, metadatas=metadatas)

        self._rebuild_bm25()

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
        # Use instance defaults if not specified
        k = k if k is not None else self.top_k
        alpha = alpha if alpha is not None else self.alpha
        threshold = threshold if threshold is not None else self.threshold

        # Combine default servers with client servers
        allowed_servers = self.default_allowed_servers
        if client_servers:
            allowed_servers = allowed_servers | set(client_servers)

        # Encode query
        q_vec = self._encode(query)

        # Dense (semantic) - rank-based scoring with actual similarities and metadata
        dense_ids, _, dense_metadatas = self.store.search_with_scores(
            q_vec, k, allowed_servers=allowed_servers
        )
        dense_scores = {t: 1.0 - i / k for i, t in enumerate(dense_ids)}

        # Create lookup dictionary from dense results
        metadata_lookup = {
            tool_id: json.loads(meta["tool_json"])
            for tool_id, meta in zip(dense_ids, dense_metadatas)
        }

        # Sparse (BM25) - normalized to 0-1 range
        sparse_scores, sparse_metadata = self._retrieve_sparse_scores(
            query, allowed_servers=allowed_servers
        )
        sparse_ids = sorted(sparse_scores, key=sparse_scores.get, reverse=True)[:k]

        # Merge sparse metadata into lookup so sparse-only hits have metadata
        for tool_id, tool_dict in sparse_metadata.items():
            if tool_id not in metadata_lookup:
                metadata_lookup[tool_id] = tool_dict

        # Fusion (alpha * dense + (1-alpha) * sparse)
        fused = {}
        for t in set(dense_ids + sparse_ids):
            d = dense_scores.get(t, 0)
            s = sparse_scores.get(t, 0)
            fused[t] = alpha * d + (1 - alpha) * s
        final = sorted(fused, key=fused.get, reverse=True)[:k]

        # Group tools by server, thresholding on the fused score
        server_tools: dict[str, list[dict[str, Any]]] = {}

        for name in final:
            if fused[name] < threshold:
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

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from store documents."""
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
