"""Base class for hybrid (dense + sparse) RAG retrieval."""

import re
import uuid
from collections.abc import Callable
from typing import Any

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

_NON_ALPHA = re.compile(r"[^a-z0-9\s]")

# Subset of NLTK's English stop-word list, inlined to avoid the dependency.
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "we",
        "you",
        "he",
        "she",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "our",
        "your",
        "his",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
        "than",
        "too",
        "very",
        "just",
        "about",
        "above",
        "after",
        "before",
        "between",
        "into",
        "out",
        "up",
        "down",
        "over",
        "under",
        "again",
        "further",
        "once",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop words, and split."""
    tokens = _NON_ALPHA.sub("", text.lower()).split()
    return [t for t in tokens if t not in _STOP_WORDS]


class QdrantStore:
    """Wrapper for in-memory vector database operations backed by Qdrant."""

    def __init__(self, collection: str) -> None:
        """Initialize in-memory Qdrant client.

        Args:
            collection: Name of the Qdrant collection.
        """
        self._collection = collection
        self.client = QdrantClient(location=":memory:")
        self._collection_ready = False

    def _ensure_collection(self, vector_size: int) -> None:
        """Lazily create the vector collection on first upsert."""
        if self._collection_ready:
            return
        self.client.create_collection(
            self._collection,
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
            ids: List of unique identifiers for documents.
            docs: List of document texts.
            vectors: List of embedding vectors.
            metadatas: Optional list of metadata dictionaries.
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
        self.client.upsert(self._collection, points=points)

    def search_with_scores(
        self,
        vector: list[float],
        k: int,
        allowed_servers: set[str] | None = None,
    ) -> tuple[list[str], list[float], list[dict]]:
        """Search and return IDs, similarity scores, and metadata.

        Args:
            vector: Query embedding vector.
            k: Number of results to return.
            allowed_servers: Optional set of server names to filter by.

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
            self._collection,
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
            ids: List of document IDs to delete.
        """
        if not self._collection_ready:
            return

        point_ids = [self._point_id(str_id) for str_id in ids]
        self.client.delete(
            self._collection,
            points_selector=PointIdsList(points=point_ids),
        )

    def get_all(self) -> dict:
        """Get all documents with their metadata.

        Returns:
            Dictionary with 'ids', 'documents', and 'metadatas' keys.
        """
        if not self._collection_ready:
            return {"ids": [], "documents": [], "metadatas": []}

        records, _ = self.client.scroll(self._collection, limit=10_000)

        out_ids: list[str] = []
        documents: list[str] = []
        metas: list[dict] = []
        for record in records:
            payload = record.payload or {}
            out_ids.append(payload["_id"])
            documents.append(payload.get("_document", ""))
            metas.append({k: v for k, v in payload.items() if not k.startswith("_")})
        return {"ids": out_ids, "documents": documents, "metadatas": metas}


class HybridRAGBase:
    """Base class for hybrid retrieval using dense (Qdrant) and sparse (BM25) methods.

    Subclasses implement domain-specific indexing and result formatting
    while reusing the shared retrieval algorithm.
    """

    def __init__(
        self,
        collection: str,
        encode_fn: Callable[[str], list[float]],
        alpha: float = 0.8,
        top_k: int = 10,
        threshold: float = 0.01,
    ) -> None:
        """Initialize the hybrid RAG system.

        Args:
            collection: Name of the Qdrant collection (provided by subclass).
            encode_fn: Function that encodes text into an embedding vector.
            alpha: Weight for dense vs sparse (1.0 = full dense, 0.0 = full sparse).
            top_k: Number of results to retrieve.
            threshold: Minimum similarity threshold for filtering results.
        """
        self.alpha = alpha
        self.top_k = top_k
        self.threshold = threshold
        self._encode = encode_fn
        self.bm25: BM25Okapi | None = None
        self.store = QdrantStore(collection)

    def _index_documents(
        self,
        ids: list[str],
        docs: list[str],
        vectors: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Index documents in the store and rebuild BM25.

        Args:
            ids: Document identifiers.
            docs: Document texts (used for both dense and sparse retrieval).
            vectors: Pre-computed embedding vectors.
            metadatas: Optional metadata dicts for each document.
        """
        self.store.upsert(ids, docs, vectors, metadatas=metadatas)
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from all stored documents."""
        all_data = self.store.get_all()
        if not all_data["documents"]:
            self.bm25 = None
            return
        sparse_docs = [_tokenize(doc) for doc in all_data["documents"]]
        self.bm25 = BM25Okapi(sparse_docs)

    def _dense_scores(
        self,
        query_vec: list[float],
        k: int,
        allowed_servers: set[str] | None = None,
    ) -> tuple[dict[str, float], list[str], list[dict]]:
        """Compute dense retrieval scores using cosine similarity from Qdrant.

        Args:
            query_vec: Query embedding vector.
            k: Number of results.
            allowed_servers: Optional server filter (passed to QdrantStore).

        Returns:
            Tuple of (id-to-score dict, ordered id list, metadata list).
        """
        ids, sim_scores, metas = self.store.search_with_scores(
            query_vec, k, allowed_servers=allowed_servers
        )
        scores = dict(zip(ids, sim_scores))
        return scores, ids, metas

    def _sparse_scores(self, query: str) -> dict[str, float]:
        """Compute BM25 scores normalized to 0-1 range.

        Args:
            query: The query string.

        Returns:
            Dict mapping document IDs to normalized BM25 scores.
        """
        if self.bm25 is None:
            return {}

        all_data = self.store.get_all()
        ids = all_data["ids"]
        raw = self.bm25.get_scores(_tokenize(query))
        # BM25 IDF is negative when corpus has few documents (log((n-df+0.5)/(df+0.5)) < 0
        # when df >= n), which would penalize the fused score. Clamp to zero.
        clamped = [max(0.0, s) for s in raw]
        mx = max(clamped) if max(clamped) > 0 else 1.0
        # Normalize to [0, 1] so sparse scores are comparable to dense cosine similarity.
        return {sid: score / mx for sid, score in zip(ids, clamped)}

    @staticmethod
    def _fuse_scores(
        dense: dict[str, float],
        sparse: dict[str, float],
        alpha: float,
        k: int,
    ) -> dict[str, float]:
        """Fuse dense and sparse scores using weighted combination.

        Args:
            dense: Dense retrieval scores (id → score).
            sparse: Sparse retrieval scores (id → score).
            alpha: Weight for dense (1-alpha for sparse).
            k: Maximum number of results.

        Returns:
            Dict mapping IDs to fused scores, top k, sorted descending.
        """
        # Weighted linear combination: alpha=1.0 is pure semantic, alpha=0.0 is pure keyword.
        # Documents appearing in only one set get 0 for the missing component.
        fused: dict[str, float] = {}
        for t in set(list(dense.keys()) + list(sparse.keys())):
            d = dense.get(t, 0)
            s = sparse.get(t, 0)
            fused[t] = alpha * d + (1 - alpha) * s
        return dict(sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k])
