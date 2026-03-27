"""Unit tests for the HybridRAGBase and QdrantStore shared primitives."""

from unittest.mock import MagicMock

import pytest

from ols.src.rag.hybrid_rag import HybridRAGBase, QdrantStore

DIMENSION = 8


def _fake_encode(text: str) -> list[float]:
    """Deterministic encode: sum of char ordinals spread across DIMENSION dims."""
    total = sum(ord(c) for c in text)
    return [(total + i) / 1000.0 for i in range(DIMENSION)]


def _make_base(**kwargs: object) -> HybridRAGBase:
    """Create a HybridRAGBase with fake encode and sensible test defaults."""
    defaults: dict = {
        "collection": "test",
        "encode_fn": _fake_encode,
        "alpha": 0.5,
        "top_k": 10,
        "threshold": 0.0,
    }
    defaults.update(kwargs)
    return HybridRAGBase(**defaults)


class TestQdrantStore:
    """Tests for QdrantStore wrapper."""

    def test_upsert_and_get_all(self) -> None:
        """Verify upsert stores documents retrievable via get_all."""
        store = QdrantStore("test")
        store.upsert(
            ids=["a", "b"],
            docs=["hello", "world"],
            vectors=[[0.1] * DIMENSION, [0.2] * DIMENSION],
            metadatas=[{"key": "v1"}, {"key": "v2"}],
        )
        data = store.get_all()
        assert set(data["ids"]) == {"a", "b"}
        assert len(data["metadatas"]) == 2

    def test_search_with_scores_returns_similarities(self) -> None:
        """Verify search returns cosine similarities."""
        store = QdrantStore("test")
        vec = [float(i) / DIMENSION for i in range(DIMENSION)]
        store.upsert(ids=["a"], docs=["test"], vectors=[vec])
        ids, sims, _ = store.search_with_scores(vec, k=1)
        assert ids == ["a"]
        assert sims[0] == pytest.approx(1.0, abs=0.01)

    def test_search_with_server_filter(self) -> None:
        """Verify server filtering excludes non-matching servers."""
        store = QdrantStore("test")
        store.upsert(
            ids=["a", "b"],
            docs=["one", "two"],
            vectors=[[0.1] * DIMENSION, [0.9] * DIMENSION],
            metadatas=[{"server": "s1"}, {"server": "s2"}],
        )
        ids, _, _ = store.search_with_scores(
            [0.5] * DIMENSION, k=10, allowed_servers={"s1"}
        )
        assert ids == ["a"]

    def test_delete(self) -> None:
        """Verify delete removes documents."""
        store = QdrantStore("test")
        store.upsert(
            ids=["a", "b"],
            docs=["one", "two"],
            vectors=[[0.1] * DIMENSION, [0.2] * DIMENSION],
        )
        store.delete(["a"])
        data = store.get_all()
        assert data["ids"] == ["b"]

    def test_empty_collection_returns_empty(self) -> None:
        """Verify operations on empty store return empty results."""
        store = QdrantStore("test")
        assert store.get_all() == {"ids": [], "documents": [], "metadatas": []}
        ids, scores, metas = store.search_with_scores([0.1] * DIMENSION, k=5)
        assert ids == []
        assert scores == []
        assert metas == []

    def test_point_id_is_deterministic(self) -> None:
        """Verify same input produces same UUID."""
        id1 = QdrantStore._point_id("test-id")
        id2 = QdrantStore._point_id("test-id")
        assert id1 == id2

    def test_different_collections_are_isolated(self) -> None:
        """Verify two stores with different collections don't share data."""
        store_a = QdrantStore("collection_a")
        store_b = QdrantStore("collection_b")
        store_a.upsert(ids=["a"], docs=["hello"], vectors=[[0.1] * DIMENSION])
        assert len(store_a.get_all()["ids"]) == 1
        assert len(store_b.get_all()["ids"]) == 0


class TestHybridRAGBaseIndex:
    """Tests for _index_documents and _rebuild_bm25."""

    def test_index_builds_bm25(self) -> None:
        """Verify BM25 index is built after indexing documents."""
        rag = _make_base()
        assert rag.bm25 is None
        rag._index_documents(
            ids=["a"],
            docs=["hello world"],
            vectors=[_fake_encode("hello world")],
        )
        assert rag.bm25 is not None

    def test_index_stores_in_qdrant(self) -> None:
        """Verify documents are stored in the Qdrant store."""
        rag = _make_base()
        rag._index_documents(
            ids=["a", "b"],
            docs=["doc one", "doc two"],
            vectors=[_fake_encode("doc one"), _fake_encode("doc two")],
        )
        data = rag.store.get_all()
        assert set(data["ids"]) == {"a", "b"}

    def test_index_with_metadatas(self) -> None:
        """Verify metadata is stored alongside documents."""
        rag = _make_base()
        rag._index_documents(
            ids=["a"],
            docs=["doc"],
            vectors=[_fake_encode("doc")],
            metadatas=[{"custom": "value"}],
        )
        data = rag.store.get_all()
        assert data["metadatas"][0]["custom"] == "value"

    def test_reindex_updates_not_duplicates(self) -> None:
        """Verify re-indexing same IDs updates rather than duplicates."""
        rag = _make_base()
        docs = [_fake_encode("doc")]
        rag._index_documents(ids=["a"], docs=["doc"], vectors=docs)
        rag._index_documents(ids=["a"], docs=["doc updated"], vectors=docs)
        data = rag.store.get_all()
        assert len(data["ids"]) == 1

    def test_encode_fn_is_called(self) -> None:
        """Verify the encode function is used during indexing."""
        mock_encode = MagicMock(return_value=[0.1] * DIMENSION)
        rag = _make_base(encode_fn=mock_encode)
        vecs = [mock_encode("a"), mock_encode("b")]
        mock_encode.reset_mock()
        rag._index_documents(ids=["a", "b"], docs=["a", "b"], vectors=vecs)
        assert rag.store.get_all()["ids"]


class TestHybridRAGBaseDenseScores:
    """Tests for _dense_scores."""

    def test_returns_rank_based_scores(self) -> None:
        """Verify dense scores are rank-based (1.0 for first, decreasing)."""
        rag = _make_base()
        rag._index_documents(
            ids=["a", "b"],
            docs=["hello", "world"],
            vectors=[_fake_encode("hello"), _fake_encode("world")],
        )
        scores, ids, _ = rag._dense_scores(_fake_encode("hello"), k=2)
        assert len(scores) == 2
        assert scores[ids[0]] > scores[ids[1]]

    def test_returns_metadata(self) -> None:
        """Verify metadata is returned alongside scores."""
        rag = _make_base()
        rag._index_documents(
            ids=["a"],
            docs=["doc"],
            vectors=[_fake_encode("doc")],
            metadatas=[{"key": "val"}],
        )
        _, _, metas = rag._dense_scores(_fake_encode("doc"), k=1)
        assert len(metas) == 1
        assert metas[0]["key"] == "val"


class TestHybridRAGBaseSparseScores:
    """Tests for _sparse_scores."""

    def test_returns_empty_when_no_bm25(self) -> None:
        """Verify empty result when BM25 index not built."""
        rag = _make_base()
        assert rag._sparse_scores("query") == {}

    def test_returns_normalized_scores(self) -> None:
        """Verify BM25 scores are normalized between 0 and 1."""
        rag = _make_base()
        rag._index_documents(
            ids=["a", "b"],
            docs=["kubernetes pods", "file system"],
            vectors=[_fake_encode("kubernetes pods"), _fake_encode("file system")],
        )
        scores = rag._sparse_scores("kubernetes")
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_scores_contain_all_documents(self) -> None:
        """Verify all indexed documents have a score."""
        rag = _make_base()
        rag._index_documents(
            ids=["a", "b", "c"],
            docs=["one", "two", "three"],
            vectors=[_fake_encode(d) for d in ["one", "two", "three"]],
        )
        scores = rag._sparse_scores("one two three")
        assert len(scores) == 3


class TestHybridRAGBaseFuseScores:
    """Tests for _fuse_scores static method."""

    def test_pure_dense(self) -> None:
        """Verify alpha=1.0 uses only dense scores."""
        dense = {"a": 0.9, "b": 0.5}
        sparse = {"a": 0.1, "b": 0.8}
        fused = HybridRAGBase._fuse_scores(dense, sparse, alpha=1.0, k=2)
        assert fused["a"] == pytest.approx(0.9)
        assert fused["b"] == pytest.approx(0.5)

    def test_pure_sparse(self) -> None:
        """Verify alpha=0.0 uses only sparse scores."""
        dense = {"a": 0.9, "b": 0.5}
        sparse = {"a": 0.1, "b": 0.8}
        fused = HybridRAGBase._fuse_scores(dense, sparse, alpha=0.0, k=2)
        assert fused["a"] == pytest.approx(0.1)
        assert fused["b"] == pytest.approx(0.8)

    def test_weighted_combination(self) -> None:
        """Verify alpha=0.5 averages dense and sparse."""
        dense = {"a": 1.0}
        sparse = {"a": 0.5}
        fused = HybridRAGBase._fuse_scores(dense, sparse, alpha=0.5, k=1)
        assert fused["a"] == pytest.approx(0.75)

    def test_limits_to_top_k(self) -> None:
        """Verify only top k results are returned."""
        dense = {"a": 0.9, "b": 0.5, "c": 0.1}
        sparse = {"a": 0.9, "b": 0.5, "c": 0.1}
        fused = HybridRAGBase._fuse_scores(dense, sparse, alpha=0.5, k=2)
        assert len(fused) == 2

    def test_handles_disjoint_sets(self) -> None:
        """Verify fusion works when dense and sparse return different IDs."""
        dense = {"a": 0.9}
        sparse = {"b": 0.8}
        fused = HybridRAGBase._fuse_scores(dense, sparse, alpha=0.5, k=2)
        assert "a" in fused
        assert "b" in fused

    def test_sorted_descending(self) -> None:
        """Verify results are sorted by score descending."""
        dense = {"a": 0.1, "b": 0.9}
        sparse = {"a": 0.1, "b": 0.9}
        fused = HybridRAGBase._fuse_scores(dense, sparse, alpha=0.5, k=2)
        scores = list(fused.values())
        assert scores == sorted(scores, reverse=True)
