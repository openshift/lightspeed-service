"""Unit tests for hybrid_tools_rag module."""

from unittest.mock import MagicMock

import pytest

from ols.src.tools.tools_rag.hybrid_tools_rag import QdrantStore, ToolsRAG

DIMENSION = 8


def _fake_encode(text: str) -> list[float]:
    """Deterministic encode: sum of char ordinals spread across DIMENSION dims."""
    total = sum(ord(c) for c in text)
    return [(total + i) / 1000.0 for i in range(DIMENSION)]


def _make_tool(name: str, description: str, server: str) -> MagicMock:
    """Create a mock tool with the attributes ToolsRAG expects."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.args_schema = None
    tool.metadata = {"mcp_server": server}
    return tool


def _make_rag(**kwargs) -> ToolsRAG:
    """Create a ToolsRAG with fake encode and sensible test defaults."""
    defaults = {
        "encode_fn": _fake_encode,
        "alpha": 0.5,
        "top_k": 10,
        "threshold": 0.0,
    }
    defaults.update(kwargs)
    return ToolsRAG(**defaults)


def _sample_tools() -> list[MagicMock]:
    """Return a fixed set of tools spanning two servers."""
    return [
        _make_tool("get_pods", "List Kubernetes pods in a namespace", "k8s-server"),
        _make_tool("get_namespaces", "List Kubernetes namespaces", "k8s-server"),
        _make_tool("read_file", "Read contents of a file", "file-server"),
        _make_tool("search_files", "Search for files by pattern", "file-server"),
    ]


class TestQdrantStore:
    """Tests for QdrantStore wrapper."""

    def test_upsert_and_get_all(self) -> None:
        """Verify upsert stores documents retrievable via get_all."""
        store = QdrantStore()
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
        """Verify search returns cosine similarities (1 - distance)."""
        store = QdrantStore()
        vec = [float(i) / DIMENSION for i in range(DIMENSION)]
        store.upsert(
            ids=["a"],
            docs=["test"],
            vectors=[vec],
            metadatas=[{"server": "s1"}],
        )
        ids, sims, _metas = store.search_with_scores(vec, k=1)
        assert ids == ["a"]
        assert sims[0] == pytest.approx(1.0, abs=0.01)

    def test_search_with_server_filter(self) -> None:
        """Verify server filtering excludes non-matching servers."""
        store = QdrantStore()
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
        store = QdrantStore()
        store.upsert(
            ids=["a", "b"],
            docs=["one", "two"],
            vectors=[[0.1] * DIMENSION, [0.2] * DIMENSION],
        )
        store.delete(["a"])
        data = store.get_all()
        assert data["ids"] == ["b"]


class TestToolsRAGPopulate:
    """Tests for ToolsRAG.populate_tools."""

    def test_populate_stores_tools_in_qdrant(self) -> None:
        """Verify populate_tools stores all tools with correct IDs and metadata."""
        rag = _make_rag()
        tools = _sample_tools()
        rag.populate_tools(tools)

        data = rag.store.get_all()
        assert len(data["ids"]) == 4
        assert "k8s-server::get_pods" in data["ids"]
        assert "file-server::read_file" in data["ids"]

    def test_populate_builds_bm25_index(self) -> None:
        """Verify BM25 index is built after population."""
        rag = _make_rag()
        assert rag.bm25 is None
        rag.populate_tools(_sample_tools())
        assert rag.bm25 is not None

    def test_populate_upsert_updates_existing(self) -> None:
        """Verify re-populating with same tools updates rather than duplicates."""
        rag = _make_rag()
        tools = _sample_tools()
        rag.populate_tools(tools)
        rag.populate_tools(tools)

        data = rag.store.get_all()
        assert len(data["ids"]) == 4

    def test_populate_calls_encode_fn(self) -> None:
        """Verify the encode function is called for each tool."""
        mock_encode = MagicMock(return_value=[0.1] * DIMENSION)
        rag = _make_rag(encode_fn=mock_encode)
        rag.populate_tools(_sample_tools())
        assert mock_encode.call_count == 4


class TestToolsRAGRetrieveHybrid:
    """Tests for ToolsRAG.retrieve_hybrid."""

    def _populated_rag(self, **kwargs) -> ToolsRAG:
        """Create and populate a ToolsRAG with sample tools."""
        rag = _make_rag(**kwargs)
        rag.set_default_servers(["k8s-server", "file-server"])
        rag.populate_tools(_sample_tools())
        return rag

    def test_retrieve_returns_server_grouped_results(self) -> None:
        """Verify results are grouped by server name."""
        rag = self._populated_rag()
        result = rag.retrieve_hybrid("list pods")
        assert isinstance(result, dict)
        for server_name, tools in result.items():
            assert isinstance(server_name, str)
            assert isinstance(tools, list)
            for tool in tools:
                assert "server" not in tool

    def test_retrieve_respects_top_k(self) -> None:
        """Verify at most top_k tools are returned."""
        rag = self._populated_rag(top_k=2)
        result = rag.retrieve_hybrid("kubernetes pods files")
        total_tools = sum(len(t) for t in result.values())
        assert total_tools <= 2

    def test_retrieve_threshold_filters_low_scores(self) -> None:
        """Verify threshold removes low-scoring tools."""
        rag = self._populated_rag(threshold=0.99)
        result = rag.retrieve_hybrid("kubernetes pods")
        total_tools = sum(len(t) for t in result.values())
        assert total_tools == 0

    def test_retrieve_uses_override_params(self) -> None:
        """Verify k, alpha, threshold overrides work."""
        rag = self._populated_rag(top_k=10, threshold=0.0)
        result = rag.retrieve_hybrid("list pods", k=1, alpha=1.0, threshold=0.0)
        total_tools = sum(len(t) for t in result.values())
        assert total_tools <= 1

    def test_retrieve_with_no_tools_returns_empty(self) -> None:
        """Verify ToolsRAG with no matching tools returns empty results."""
        rag = self._populated_rag(threshold=0.99)
        result = rag.retrieve_hybrid("completely unrelated xyz", threshold=0.99)
        assert result == {}


class TestToolsRAGSparseOnlyHits:
    """Tests for sparse-only hits being included (bug fix coverage)."""

    def test_alpha_zero_uses_sparse_only(self) -> None:
        """Verify alpha=0.0 (full sparse) returns BM25-matched tools."""
        rag = _make_rag(alpha=0.0, threshold=0.0, top_k=10)
        rag.set_default_servers(["k8s-server", "file-server"])
        rag.populate_tools(_sample_tools())

        result = rag.retrieve_hybrid("pods", alpha=0.0)
        total_tools = sum(len(t) for t in result.values())
        assert total_tools > 0

    def test_alpha_one_uses_dense_only(self) -> None:
        """Verify alpha=1.0 (full dense) still returns results."""
        rag = _make_rag(alpha=1.0, threshold=0.0, top_k=10)
        rag.set_default_servers(["k8s-server", "file-server"])
        rag.populate_tools(_sample_tools())

        result = rag.retrieve_hybrid("pods", alpha=1.0)
        total_tools = sum(len(t) for t in result.values())
        assert total_tools > 0

    def test_sparse_only_hit_has_metadata(self) -> None:
        """Verify a tool found only by BM25 still has its metadata populated."""
        call_count = {"n": 0}

        def encode_that_varies(text: str) -> list[float]:
            call_count["n"] += 1
            return [
                float(call_count["n"]) / 100.0 + i / 1000.0 for i in range(DIMENSION)
            ]

        rag = _make_rag(
            encode_fn=encode_that_varies, alpha=0.0, threshold=0.0, top_k=10
        )
        rag.set_default_servers(["k8s-server", "file-server"])
        rag.populate_tools(_sample_tools())

        result = rag.retrieve_hybrid("pods namespaces", alpha=0.0)
        all_tools = [t for tools in result.values() for t in tools]
        for tool in all_tools:
            assert "name" in tool
            assert "desc" in tool


class TestRetrieveSparseScores:
    """Tests for _retrieve_sparse_scores."""

    def test_returns_empty_when_no_bm25(self) -> None:
        """Verify empty result when BM25 index not built."""
        rag = _make_rag()
        scores, metadata = rag._retrieve_sparse_scores("query")
        assert scores == {}
        assert metadata == {}

    def test_returns_scores_and_metadata(self) -> None:
        """Verify scores and metadata are returned for populated tools."""
        rag = _make_rag()
        rag.populate_tools(_sample_tools())

        scores, metadata = rag._retrieve_sparse_scores("kubernetes")
        assert len(scores) > 0
        assert len(metadata) > 0
        for tool_id in scores:
            assert tool_id in metadata
            assert "name" in metadata[tool_id]

    def test_server_filtering(self) -> None:
        """Verify server filtering excludes tools from non-allowed servers."""
        rag = _make_rag()
        rag.populate_tools(_sample_tools())

        scores, metadata = rag._retrieve_sparse_scores(
            "files", allowed_servers={"file-server"}
        )
        for tool_id in scores:
            assert metadata[tool_id].get("server") == "file-server"

    def test_scores_normalized_to_zero_one(self) -> None:
        """Verify BM25 scores are normalized between 0 and 1."""
        rag = _make_rag()
        rag.set_default_servers(["k8s-server", "file-server"])
        rag.populate_tools(_sample_tools())

        scores, _ = rag._retrieve_sparse_scores("pods")
        for score in scores.values():
            assert 0.0 <= score <= 1.0


class TestServerFiltering:
    """Tests for server filtering and set_default_servers."""

    def test_set_default_servers(self) -> None:
        """Verify set_default_servers updates the allowed set."""
        rag = _make_rag()
        assert rag.default_allowed_servers == set()
        rag.set_default_servers(["s1", "s2"])
        assert rag.default_allowed_servers == {"s1", "s2"}

    def test_retrieve_combines_default_and_client_servers(self) -> None:
        """Verify client_servers are merged with defaults."""
        rag = _make_rag(threshold=0.0)
        tools = [
            _make_tool("t1", "tool one", "default-server"),
            _make_tool("t2", "tool two", "client-server"),
        ]
        rag.set_default_servers(["default-server"])
        rag.populate_tools(tools)

        result_without = rag.retrieve_hybrid("tool", client_servers=None)
        result_with = rag.retrieve_hybrid("tool", client_servers=["client-server"])

        assert "default-server" in result_with
        assert "client-server" in result_with
        assert "client-server" not in result_without

    def test_retrieve_filters_to_default_servers_only(self) -> None:
        """Verify only default server tools are returned when no client servers."""
        rag = _make_rag(threshold=0.0)
        tools = [
            _make_tool("t1", "tool one", "s1"),
            _make_tool("t2", "tool two", "s2"),
        ]
        rag.set_default_servers(["s1"])
        rag.populate_tools(tools)

        result = rag.retrieve_hybrid("tool")
        assert "s1" in result
        assert "s2" not in result


class TestRemoveTools:
    """Tests for ToolsRAG.remove_tools."""

    def test_remove_deletes_from_store(self) -> None:
        """Verify remove_tools deletes specified tools."""
        rag = _make_rag()
        rag.set_default_servers(["k8s-server", "file-server"])
        rag.populate_tools(_sample_tools())

        initial_count = len(rag.store.get_all()["ids"])
        rag.remove_tools(["k8s-server::get_pods"])

        data = rag.store.get_all()
        assert len(data["ids"]) == initial_count - 1
        assert "k8s-server::get_pods" not in data["ids"]

    def test_remove_rebuilds_bm25(self) -> None:
        """Verify BM25 is rebuilt after removal."""
        rag = _make_rag()
        rag.set_default_servers(["k8s-server"])
        rag.populate_tools([_make_tool("t1", "desc", "k8s-server")])
        assert rag.bm25 is not None

        rag.remove_tools(["k8s-server::t1"])
        assert rag.bm25 is None


class TestConvertLangchainTool:
    """Tests for _convert_langchain_tool_to_dict."""

    def test_converts_tool_to_dict(self) -> None:
        """Verify correct dict structure from StructuredTool."""
        rag = _make_rag()
        tool = _make_tool("my_tool", "Does stuff", "my-server")
        result = rag._convert_langchain_tool_to_dict(tool)

        assert result["name"] == "my_tool"
        assert result["desc"] == "Does stuff"
        assert result["server"] == "my-server"

    def test_handles_missing_metadata(self) -> None:
        """Verify graceful handling when metadata is absent."""
        rag = _make_rag()
        tool = MagicMock(spec=["name", "description", "args_schema"])
        tool.name = "bare"
        tool.description = "no meta"
        tool.args_schema = None
        result = rag._convert_langchain_tool_to_dict(tool)

        assert result["name"] == "bare"
        assert result["server"] is None

    def test_serializes_args_schema_from_model(self) -> None:
        """Verify Pydantic args_schema is converted to JSON schema dict."""
        from pydantic import BaseModel, Field  # pylint: disable=import-outside-toplevel

        class MyArgs(BaseModel):
            namespace: str = Field(description="The namespace")

        rag = _make_rag()
        tool = _make_tool("my_tool", "Does stuff", "my-server")
        tool.args_schema = MyArgs
        result = rag._convert_langchain_tool_to_dict(tool)

        assert isinstance(result["params"], dict)
        assert "properties" in result["params"]
        assert "namespace" in result["params"]["properties"]

    def test_passes_through_dict_args_schema(self) -> None:
        """Verify dict args_schema (from MCP tools) is passed through as-is."""
        rag = _make_rag()
        tool = _make_tool("my_tool", "Does stuff", "my-server")
        tool.args_schema = {"type": "object", "properties": {"msg": {"type": "string"}}}
        result = rag._convert_langchain_tool_to_dict(tool)

        assert result["params"] == {
            "type": "object",
            "properties": {"msg": {"type": "string"}},
        }


class TestBuildText:
    """Tests for _build_text."""

    def test_combines_name_and_desc(self) -> None:
        """Verify text representation is name + desc."""
        rag = _make_rag()
        text = rag._build_text({"name": "get_pods", "desc": "List pods"})
        assert text == "get_pods List pods"
