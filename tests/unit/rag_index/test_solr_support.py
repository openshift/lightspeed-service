"""Unit tests for Solr hybrid RAG support helpers."""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from ols.app.models.config import SolrHybridSettings
from ols.constants import RAG_SIMILARITY_CUTOFF
from ols.src.rag_index.solr_support import (
    RetrievedChunk,
    SolrHybridSearch,
    _retrieved_chunks_from_hybrid_raw_docs,
    _retrieved_chunks_from_solr_docs,
    _safe_solr_score,
    append_openshift_docs_tool_if_configured,
    dedupe_retrieved_chunks_by_parent,
    get_openshift_docs_tool,
    nodes_from_solr_retrieved_chunks,
    normalize_solr_hybrid_query,
    solr_hybrid_openshift_docs_tool_active,
    solr_hybrid_openshift_docs_tool_uses_client_lookup,
)


def test_normalize_solr_hybrid_query_strips_common_stopwords() -> None:
    """Remove common English stopwords from the query string."""
    out = normalize_solr_hybrid_query("what is the OpenShift route")
    assert "what" not in out.split()
    assert "the" not in out.split()
    assert "OpenShift" in out or "openshift" in out.lower()


def test_normalize_solr_hybrid_query_empty_fallback() -> None:
    """Return the original query when every token is a stopword."""
    out = normalize_solr_hybrid_query("the a an")
    assert out == "the a an"


def test_retrieved_chunks_from_solr_docs_maps_metadata() -> None:
    """Map Solr docs to RetrievedChunk text, score, and citation metadata."""
    docs = [
        {
            "id": "doc1",
            "title": "T1",
            "chunk": "<p>Hello</p>",
            "score": 1.5,
            "resourceName": "/foo/bar",
        }
    ]
    chunks = _retrieved_chunks_from_solr_docs(docs, content_field="chunk")
    assert len(chunks) == 1
    assert isinstance(chunks[0], RetrievedChunk)
    assert "Hello" in chunks[0].text
    assert chunks[0].score == 1.5
    assert chunks[0].metadata["title"] == "T1"
    assert chunks[0].metadata["index_origin"] == "solr_hybrid"
    assert "access.redhat.com" in chunks[0].metadata["docs_url"]


def test_nodes_from_solr_retrieved_chunks_sets_origin_and_score() -> None:
    """Solr passages become LlamaIndex nodes with stable index_origin and usable scores."""
    nodes = nodes_from_solr_retrieved_chunks(
        [
            RetrievedChunk(
                text="chunk body",
                score=0.05,
                metadata={"title": "Routes", "docs_url": "https://docs.example/r"},
            )
        ]
    )
    assert len(nodes) == 1
    assert nodes[0].node.metadata["index_origin"] == "solr_hybrid"
    assert float(nodes[0].get_score() or 0) >= RAG_SIMILARITY_CUTOFF


def test_solr_hybrid_openshift_docs_tool_uses_client_lookup() -> None:
    """Resolve Solr client for the docs tool only when hybrid is tool-only (not direct RAG)."""
    assert solr_hybrid_openshift_docs_tool_uses_client_lookup(None) is False
    assert (
        solr_hybrid_openshift_docs_tool_uses_client_lookup(
            SolrHybridSettings(solr_direct_rag=True)
        )
        is False
    )
    assert (
        solr_hybrid_openshift_docs_tool_uses_client_lookup(
            SolrHybridSettings(solr_direct_rag=False)
        )
        is True
    )


def test_solr_hybrid_openshift_docs_tool_active_requires_client() -> None:
    """Active docs tool requires tool-only settings and a non-``None`` client."""
    cfg = SolrHybridSettings(solr_direct_rag=False)
    assert solr_hybrid_openshift_docs_tool_active(cfg, None) is False
    assert solr_hybrid_openshift_docs_tool_active(cfg, MagicMock()) is True
    cfg_direct = SolrHybridSettings(solr_direct_rag=True)
    assert solr_hybrid_openshift_docs_tool_active(cfg_direct, MagicMock()) is False


def test_append_openshift_docs_tool_if_configured_appends_when_active() -> None:
    """Append one tool when settings and client satisfy the tool-only policy."""
    tools: list = []
    cfg = SolrHybridSettings(solr_direct_rag=False)
    append_openshift_docs_tool_if_configured(
        tools, solr_hybrid=cfg, solr_client=MagicMock()
    )
    assert len(tools) == 1
    assert tools[0].name == "search_openshift_documentation"


def test_append_openshift_docs_tool_if_configured_skips_when_direct_rag() -> None:
    """Do not append when direct Solr RAG is enabled (tool is not registered)."""
    tools: list = []
    cfg = SolrHybridSettings(solr_direct_rag=True)
    append_openshift_docs_tool_if_configured(
        tools, solr_hybrid=cfg, solr_client=MagicMock()
    )
    assert tools == []


def test_solr_hybrid_settings_rejects_unknown_field() -> None:
    """Reject unexpected fields on SolrHybridSettings."""
    with pytest.raises(ValidationError):
        SolrHybridSettings(unknown_field=True)  # type: ignore[call-arg]


def test_solr_hybrid_settings_solr_direct_rag_default() -> None:
    """Direct RAG is the default; tool-only mode is opt-in."""
    assert SolrHybridSettings().solr_direct_rag is True
    assert SolrHybridSettings(solr_direct_rag=False).solr_direct_rag is False


@pytest.mark.asyncio
async def test_solr_hybrid_search_invokes_encode_fn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass normalized query text through the injected embedding callable."""
    seen: list[str] = []

    def encode_fn(text: str) -> list[float]:
        seen.append(text)
        return [0.25, 0.5, 0.75]

    async def fake_post(
        url: str,
        form: dict[str, str],
        client_timeout_s: float,
    ) -> dict:
        assert "hybrid-search" in url
        assert form.get("defType") == "edismax"
        assert "chunk_vector" in form.get("rqq", "")
        assert form.get("q") == "pod"
        return {"response": {"docs": []}}

    monkeypatch.setattr(
        "ols.src.rag_index.solr_support._post_hybrid",
        fake_post,
    )
    client = SolrHybridSearch(SolrHybridSettings(), encode_fn)
    await client.search("what is the pod")
    assert seen == [normalize_solr_hybrid_query("what is the pod")]


@pytest.mark.asyncio
async def test_solr_hybrid_search_returns_empty_when_hybrid_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid-search with no docs returns ``[]`` (no lexical fallback)."""

    def encode_fn(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def fake_post(
        url: str,
        form: dict[str, str],
        client_timeout_s: float,
    ) -> dict:
        return {"response": {"docs": []}}

    monkeypatch.setattr("ols.src.rag_index.solr_support._post_hybrid", fake_post)

    client = SolrHybridSearch(SolrHybridSettings(max_results=3), encode_fn)
    chunks = await client.search("openshift routes")
    assert chunks == []


@pytest.mark.asyncio
async def test_solr_hybrid_search_prefers_hybrid_when_docs_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return hybrid chunks when hybrid-search returns docs."""

    def encode_fn(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def fake_post(
        url: str,
        form: dict[str, str],
        client_timeout_s: float,
    ) -> dict:
        return {
            "response": {
                "docs": [
                    {
                        "id": "h1",
                        "title": "Hybrid Title",
                        "chunk": "<p>hybrid only</p>",
                        "score": 1.0,
                        "resourceName": "/docs/hybrid",
                    }
                ]
            }
        }

    monkeypatch.setattr("ols.src.rag_index.solr_support._post_hybrid", fake_post)

    client = SolrHybridSearch(SolrHybridSettings(max_results=2), encode_fn)
    chunks = await client.search("routes")
    assert len(chunks) == 1
    assert "hybrid only" in chunks[0].text
    assert chunks[0].metadata["index_origin"] == "solr_hybrid"


def test_dedupe_retrieved_chunks_by_parent_keeps_first() -> None:
    """First chunk per ``parent_id`` wins; chunks without key are kept."""
    a = RetrievedChunk(text="a", metadata={"parent_id": "p1", "id": "1"})
    b = RetrievedChunk(text="b", metadata={"parent_id": "p1", "id": "2"})
    c = RetrievedChunk(text="c", metadata={"parent_id": "p2", "id": "3"})
    d = RetrievedChunk(text="d", metadata={})
    out = dedupe_retrieved_chunks_by_parent([a, b, c, d])
    assert [x.text for x in out] == ["a", "c", "d"]


@pytest.mark.asyncio
async def test_solr_hybrid_search_dedupes_and_caps_max_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parent dedupe then ``max_results`` slice across distinct parents."""

    def encode_fn(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def fake_post(
        url: str,
        form: dict[str, str],
        client_timeout_s: float,
    ) -> dict:
        return {
            "response": {
                "docs": [
                    {
                        "id": "a1",
                        "parent_id": "p1",
                        "title": "T1",
                        "chunk": "<p>first p1</p>",
                        "score": 10.0,
                        "resourceName": "/r1",
                    },
                    {
                        "id": "a2",
                        "parent_id": "p1",
                        "title": "T1b",
                        "chunk": "<p>dup p1</p>",
                        "score": 9.0,
                        "resourceName": "/r1b",
                    },
                    {
                        "id": "b1",
                        "parent_id": "p2",
                        "title": "T2",
                        "chunk": "<p>p2 only</p>",
                        "score": 8.0,
                        "resourceName": "/r2",
                    },
                    {
                        "id": "c1",
                        "parent_id": "p3",
                        "title": "T3",
                        "chunk": "<p>p3 dropped by cap</p>",
                        "score": 7.0,
                        "resourceName": "/r3",
                    },
                ]
            }
        }

    monkeypatch.setattr("ols.src.rag_index.solr_support._post_hybrid", fake_post)

    client = SolrHybridSearch(SolrHybridSettings(max_results=2), encode_fn)
    chunks = await client.search("q")
    assert len(chunks) == 2
    assert "first p1" in chunks[0].text
    assert "p2 only" in chunks[1].text


@pytest.mark.asyncio
async def test_get_openshift_docs_tool_returns_json() -> None:
    """Tool coroutine returns JSON list of passages from the bound client."""

    class _StubSolrClient:
        """Minimal async client stub matching ``SolrHybridSearch.search`` for tool tests."""

        async def search(self, query: str) -> list[RetrievedChunk]:
            assert query == "routes"
            return [
                RetrievedChunk(
                    text="body",
                    score=0.5,
                    metadata={"title": "T", "docs_url": "https://access.redhat.com/x"},
                )
            ]

    tool = get_openshift_docs_tool(_StubSolrClient())  # type: ignore[arg-type]
    out = await tool.coroutine(search_query="routes")  # type: ignore[misc]
    data = json.loads(out)
    assert len(data) == 1
    assert data[0]["text"] == "body"
    assert data[0]["score"] == 0.5
    assert data[0]["title"] == "T"
    assert "access.redhat.com" in data[0]["docs_url"]


def test_safe_solr_score_coerces_none_and_invalid() -> None:
    """Hybrid score field tolerates null and non-numeric Solr values."""
    assert _safe_solr_score(None) == 0.0
    assert _safe_solr_score("not-a-number") == 0.0
    assert _safe_solr_score(2.5) == 2.5


def test_retrieved_chunks_from_hybrid_raw_docs_null_score_no_typeerror() -> None:
    """Null ``score`` does not break threshold filtering."""
    out = _retrieved_chunks_from_hybrid_raw_docs(
        [
            {
                "id": "x",
                "title": "T",
                "chunk": "<p>body</p>",
                "score": None,
                "resourceName": "/r",
            }
        ],
        hybrid_score_threshold=0.5,
        content_field="chunk",
    )
    assert out == []


@pytest.mark.asyncio
async def test_solr_search_returns_empty_when_embedding_raises() -> None:
    """``search`` swallows embedding failures and returns no passages."""

    def bad_encode(_text: str) -> list[float]:
        raise RuntimeError("embedding unavailable")

    client = SolrHybridSearch(SolrHybridSettings(), bad_encode)
    chunks = await client.search("any query")
    assert chunks == []


@pytest.mark.asyncio
async def test_get_openshift_docs_tool_returns_error_json_when_search_raises() -> None:
    """Tool maps unexpected ``search`` raises to a structured error."""

    class _BrokenClient:
        async def search(self, _query: str) -> list[RetrievedChunk]:
            raise RuntimeError("unexpected")

    tool = get_openshift_docs_tool(_BrokenClient())  # type: ignore[arg-type]
    out = await tool.coroutine(search_query="x")  # type: ignore[misc]
    data = json.loads(out)
    assert data.get("error") == "documentation_search_failed"
