"""Unit tests for Solr hybrid RAG support helpers."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from ols.app.models.config import SolrHybridSettings
from ols.src.rag_index.solr_support import (
    RetrievedChunk,
    SolrHybridSearch,
    _safe_solr_score,
    get_openshift_docs_tool,
    normalize_solr_hybrid_query,
)
from ols.utils.checks import InvalidConfigurationError

_FAKE_REQUEST = httpx.Request("POST", "http://solr/hybrid-search")

_PATCH_RESOLVE = patch.object(
    SolrHybridSearch,
    "_resolve_chunk_filter_query",
    return_value="is_chunk:true AND product:*openshift*",
)


def _ok_response(payload: dict[str, Any]) -> httpx.Response:
    """Build an ``httpx.Response`` that supports ``raise_for_status()``."""
    return httpx.Response(200, json=payload, request=_FAKE_REQUEST)


def _patch_httpx_client(fake_post, fake_get=None):
    """Patch Solr hybrid search to use a mock persistent ``httpx.AsyncClient``."""
    mock_client = AsyncMock()
    mock_client.post = fake_post
    mock_client.get = fake_get if fake_get is not None else AsyncMock()
    mock_client.is_closed = False
    return patch.object(SolrHybridSearch, "_get_http_client", return_value=mock_client)


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


def test_get_openshift_docs_tool_has_expected_name() -> None:
    """Factory returns a tool named ``search_openshift_documentation``."""
    tool = get_openshift_docs_tool(MagicMock())
    assert tool.name == "search_openshift_documentation"


def test_solr_hybrid_settings_rejects_unknown_field() -> None:
    """Reject unexpected fields on SolrHybridSettings."""
    with pytest.raises(ValidationError):
        SolrHybridSettings(unknown_field=True)  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_solr_hybrid_search_invokes_encode_fn() -> None:
    """Pass normalized query text through the injected embedding callable."""
    seen: list[str] = []

    def encode_fn(text: str) -> list[float]:
        seen.append(text)
        return [0.25, 0.5, 0.75]

    posted: list[dict[str, str]] = []

    async def fake_post(url: str, *, data: Any, headers: Any) -> Any:
        assert "hybrid-search" in url
        assert data.get("defType") == "edismax"
        assert "chunk_vector" in data.get("rqq", "")
        assert data.get("q") == "pod"
        posted.append(data)
        resp = _ok_response({"response": {"docs": []}})
        return resp

    with _PATCH_RESOLVE:
        client = SolrHybridSearch(SolrHybridSettings(), encode_fn)
    with _patch_httpx_client(fake_post):
        await client.search("what is the pod")
    assert seen == [normalize_solr_hybrid_query("what is the pod")]


@pytest.mark.asyncio
async def test_solr_hybrid_search_returns_empty_when_hybrid_empty() -> None:
    """Hybrid-search with no docs returns ``[]`` (no lexical fallback)."""

    def encode_fn(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def fake_post(url: str, *, data: Any, headers: Any) -> Any:
        return _ok_response({"response": {"docs": []}})

    with _PATCH_RESOLVE:
        client = SolrHybridSearch(SolrHybridSettings(max_results=3), encode_fn)
    with _patch_httpx_client(fake_post):
        chunks = await client.search("openshift routes")
    assert chunks == []


@pytest.mark.asyncio
async def test_solr_hybrid_search_prefers_hybrid_when_docs_present() -> None:
    """Return hybrid chunks when hybrid-search returns docs."""

    def encode_fn(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def fake_post(url: str, *, data: Any, headers: Any) -> Any:
        return _ok_response(
            {
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
        )

    with _PATCH_RESOLVE:
        client = SolrHybridSearch(SolrHybridSettings(max_results=2), encode_fn)
    with _patch_httpx_client(fake_post):
        chunks = await client.search("routes")
    assert len(chunks) == 1
    assert "hybrid only" in chunks[0].text
    assert chunks[0].metadata["index_origin"] == "solr_hybrid"


@pytest.mark.asyncio
async def test_solr_hybrid_search_dedupes_and_caps_max_results() -> None:
    """Parent dedupe then ``max_results`` slice across distinct parents."""

    def encode_fn(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def fake_post(url: str, *, data: Any, headers: Any) -> Any:
        return _ok_response(
            {
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
            },
        )

    with _PATCH_RESOLVE:
        client = SolrHybridSearch(SolrHybridSettings(max_results=2), encode_fn)
    with _patch_httpx_client(fake_post):
        chunks = await client.search("q")
    assert len(chunks) == 2
    assert "first p1" in chunks[0].text
    assert "p2 only" in chunks[1].text


@pytest.mark.asyncio
async def test_get_openshift_docs_tool_returns_content_blocks() -> None:
    """Tool coroutine returns a list of content blocks from the bound client."""

    class _StubSolrClient:
        """Minimal async client stub matching ``SolrHybridSearch.search`` for tool tests."""

        async def search(
            self, query: str, token_budget: int = 0
        ) -> list[RetrievedChunk]:
            assert query == "routes"
            return [
                RetrievedChunk(
                    text="body",
                    score=0.5,
                    metadata={"title": "T", "docs_url": "https://access.redhat.com/x"},
                )
            ]

    tool = get_openshift_docs_tool(_StubSolrClient())  # type: ignore[arg-type]
    result = await tool.coroutine(search_query="routes")  # type: ignore[misc]
    assert isinstance(result, tuple)
    out, metadata = result
    assert len(out) == 1
    data = json.loads(out[0]["text"])
    assert data["text"] == "body"
    assert data["score"] == 0.5
    assert data["title"] == "T"
    assert "access.redhat.com" in data["docs_url"]

    ref_docs = metadata["referenced_documents"]
    assert len(ref_docs) == 1
    assert ref_docs[0].doc_url == "https://access.redhat.com/x"
    assert ref_docs[0].doc_title == "T"


@pytest.mark.asyncio
async def test_get_openshift_docs_tool_deduplicates_referenced_documents() -> None:
    """Tool deduplicates referenced_documents by URL."""

    class _MultiChunkClient:
        async def search(
            self, query: str, token_budget: int = 0
        ) -> list[RetrievedChunk]:
            return [
                RetrievedChunk(
                    text="a", score=0.9, metadata={"title": "T1", "docs_url": "u1"}
                ),
                RetrievedChunk(
                    text="b", score=0.8, metadata={"title": "T1", "docs_url": "u1"}
                ),
                RetrievedChunk(
                    text="c", score=0.7, metadata={"title": "T2", "docs_url": "u2"}
                ),
            ]

    tool = get_openshift_docs_tool(_MultiChunkClient())  # type: ignore[arg-type]
    result = await tool.coroutine(search_query="q")  # type: ignore[misc]
    out, metadata = result
    assert len(out) == 3
    ref_docs = metadata["referenced_documents"]
    assert len(ref_docs) == 2
    assert ref_docs[0].doc_url == "u1"
    assert ref_docs[1].doc_url == "u2"


def test_safe_solr_score_coerces_none_and_invalid() -> None:
    """Hybrid score field tolerates null and non-numeric Solr values."""
    assert _safe_solr_score(None) == 0.0
    assert _safe_solr_score("not-a-number") == 0.0
    assert _safe_solr_score(2.5) == 2.5


@pytest.mark.asyncio
async def test_solr_search_returns_empty_when_embedding_raises() -> None:
    """``search`` swallows embedding failures and returns no passages."""

    def bad_encode(_text: str) -> list[float]:
        raise RuntimeError("embedding unavailable")

    with _PATCH_RESOLVE:
        client = SolrHybridSearch(SolrHybridSettings(), bad_encode)
    chunks = await client.search("any query")
    assert chunks == []


@pytest.mark.asyncio
async def test_get_openshift_docs_tool_returns_error_json_when_search_raises() -> None:
    """Tool maps unexpected ``search`` raises to a structured error block."""

    class _BrokenClient:
        async def search(
            self, _query: str, token_budget: int = 0
        ) -> list[RetrievedChunk]:
            raise RuntimeError("unexpected")

    tool = get_openshift_docs_tool(_BrokenClient())  # type: ignore[arg-type]
    out = await tool.coroutine(search_query="x")  # type: ignore[misc]
    assert isinstance(out, list)
    data = json.loads(out[0]["text"])
    assert data.get("error") == "documentation_search_failed"


def test_expand_around_match_respects_token_budget() -> None:
    """Only include neighbors whose cumulative tokens fit within the budget."""
    family = [
        {"chunk_index": 0, "num_tokens": 100, "chunk": "c0"},
        {"chunk_index": 1, "num_tokens": 100, "chunk": "c1"},
        {"chunk_index": 2, "num_tokens": 100, "chunk": "c2"},
        {"chunk_index": 3, "num_tokens": 100, "chunk": "c3"},
        {"chunk_index": 4, "num_tokens": 100, "chunk": "c4"},
    ]
    result = SolrHybridSearch._expand_around_match(
        family, matched_chunk_index=2, token_budget=250
    )
    indices = [d["chunk_index"] for d in result]
    assert 2 in indices
    assert len(result) <= 3
    assert all(d["chunk_index"] in (1, 2, 3) for d in result)


def test_expand_around_match_includes_all_when_budget_is_large() -> None:
    """Include every family member when the budget exceeds the total."""
    family = [
        {"chunk_index": 0, "num_tokens": 50, "chunk": "c0"},
        {"chunk_index": 1, "num_tokens": 50, "chunk": "c1"},
        {"chunk_index": 2, "num_tokens": 50, "chunk": "c2"},
    ]
    result = SolrHybridSearch._expand_around_match(
        family, matched_chunk_index=1, token_budget=10000
    )
    assert len(result) == 3


def test_expand_around_match_zero_budget_returns_only_match() -> None:
    """Zero budget returns only the matched chunk itself."""
    family = [
        {"chunk_index": 0, "num_tokens": 50, "chunk": "c0"},
        {"chunk_index": 1, "num_tokens": 0, "chunk": "c1"},
        {"chunk_index": 2, "num_tokens": 50, "chunk": "c2"},
    ]
    result = SolrHybridSearch._expand_around_match(
        family, matched_chunk_index=1, token_budget=0
    )
    assert len(result) == 1
    assert result[0]["chunk_index"] == 1


def test_expand_around_match_caps_at_max_neighbors() -> None:
    """Even with a huge budget, expansion stops at _MAX_EXPANSION_NEIGHBORS per side."""
    family = [{"chunk_index": i, "num_tokens": 10, "chunk": f"c{i}"} for i in range(10)]
    result = SolrHybridSearch._expand_around_match(
        family, matched_chunk_index=5, token_budget=100000
    )
    assert len(result) == 5
    indices = [d["chunk_index"] for d in result]
    assert indices == [3, 4, 5, 6, 7]


@pytest.mark.asyncio
async def test_solr_search_expands_chunks_when_budget_provided() -> None:
    """With a token budget, search fetches family chunks and expands around match."""

    def encode_fn(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    family_docs = {
        "response": {
            "docs": [
                {
                    "id": "f0",
                    "parent_id": "parent1",
                    "heading_id": "h1",
                    "chunk_index": 0,
                    "num_tokens": 40,
                    "chunk": "<p>before</p>",
                },
                {
                    "id": "f1",
                    "parent_id": "parent1",
                    "heading_id": "h1",
                    "chunk_index": 1,
                    "num_tokens": 50,
                    "chunk": "<p>matched chunk</p>",
                },
                {
                    "id": "f2",
                    "parent_id": "parent1",
                    "heading_id": "h1",
                    "chunk_index": 2,
                    "num_tokens": 40,
                    "chunk": "<p>after</p>",
                },
            ]
        }
    }

    async def fake_post(url: str, *, data: Any, headers: Any) -> Any:
        return _ok_response(
            {
                "response": {
                    "docs": [
                        {
                            "id": "m1",
                            "parent_id": "parent1",
                            "heading_id": "h1",
                            "chunk_index": 1,
                            "num_tokens": 50,
                            "title": "T1",
                            "chunk": "<p>matched chunk</p>",
                            "score": 5.0,
                            "resourceName": "/docs/m",
                        }
                    ]
                }
            }
        )

    async def fake_get(url: str, *, params: Any) -> Any:
        return _ok_response(family_docs)

    mock_client = AsyncMock()
    mock_client.post = fake_post
    mock_client.get = fake_get
    mock_client.is_closed = False

    with (
        patch.object(SolrHybridSearch, "_get_http_client", return_value=mock_client),
        _PATCH_RESOLVE,
    ):
        client = SolrHybridSearch(SolrHybridSettings(max_results=3), encode_fn)
        chunks = await client.search("query", token_budget=500)
    assert len(chunks) == 1
    assert "before" in chunks[0].text
    assert "matched chunk" in chunks[0].text
    assert "after" in chunks[0].text
    assert chunks[0].metadata["chunks_expanded"] == 3


@pytest.mark.asyncio
async def test_solr_hybrid_search_max_expansion_neighbors_zero_with_budget() -> None:
    """``max_expansion_neighbors=0`` with a token budget must not raise ``NameError``."""

    def encode_fn(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def fake_post(url: str, *, data: Any, headers: Any) -> Any:
        return _ok_response(
            {
                "response": {
                    "docs": [
                        {
                            "id": "h1",
                            "title": "Hybrid Title",
                            "chunk": "<p>no expansion</p>",
                            "score": 1.0,
                            "resourceName": "/docs/hybrid",
                        }
                    ]
                }
            }
        )

    with _PATCH_RESOLVE:
        client = SolrHybridSearch(
            SolrHybridSettings(max_expansion_neighbors=0), encode_fn
        )
    with _patch_httpx_client(fake_post):
        chunks = await client.search("routes", token_budget=500)
    assert len(chunks) == 1
    assert "no expansion" in chunks[0].text


@pytest.mark.asyncio
async def test_solr_hybrid_search_reuses_async_http_client() -> None:
    """Repeated searches reuse one ``httpx.AsyncClient`` instance."""
    clients_created: list[AsyncMock] = []

    def encode_fn(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def fake_post(url: str, *, data: Any, headers: Any) -> Any:
        return _ok_response(
            {
                "response": {
                    "docs": [
                        {
                            "id": "h1",
                            "chunk": "<p>hit</p>",
                            "score": 1.0,
                            "resourceName": "/docs/hybrid",
                        }
                    ]
                }
            }
        )

    def client_factory(*args: Any, **kwargs: Any) -> AsyncMock:
        mock = AsyncMock()
        mock.post = fake_post
        mock.get = AsyncMock()
        mock.is_closed = False
        clients_created.append(mock)
        return mock

    with (
        _PATCH_RESOLVE,
        patch(
            "ols.src.rag_index.solr_support.httpx.AsyncClient",
            side_effect=client_factory,
        ),
    ):
        client = SolrHybridSearch(SolrHybridSettings(), encode_fn)
        await client.search("q1")
        await client.search("q2")
        await client.aclose()
    assert len(clients_created) == 1


@pytest.mark.asyncio
async def test_tool_passes_metadata_budget_to_search() -> None:
    """Tool reads ``tools_token_budget`` from its metadata and passes it to search."""
    received_budgets: list[int] = []

    class _BudgetCapture:
        async def search(
            self, query: str, token_budget: int = 0
        ) -> list[RetrievedChunk]:
            received_budgets.append(token_budget)
            return []

    tool = get_openshift_docs_tool(_BudgetCapture())  # type: ignore[arg-type]
    tool.metadata["tools_token_budget"] = 4096
    await tool.coroutine(search_query="test")  # type: ignore[misc]
    assert received_budgets == [4096]


def test_clamp_version_exact_match() -> None:
    """Return the exact version when it exists in available."""
    assert SolrHybridSearch._clamp_version("4.18", ["4.16", "4.18", "4.20"]) == "4.18"


def test_clamp_version_full_version_matches_major_minor() -> None:
    """Full version like 4.19.26 matches available 4.19."""
    assert (
        SolrHybridSearch._clamp_version("4.19.26", ["4.18", "4.19", "4.20"]) == "4.19"
    )


def test_clamp_version_below_range() -> None:
    """Version below all available clamps to lowest."""
    assert SolrHybridSearch._clamp_version("4.10", ["4.18", "4.19", "4.20"]) == "4.18"


def test_clamp_version_above_range() -> None:
    """Version above all available clamps to highest."""
    assert SolrHybridSearch._clamp_version("4.99", ["4.18", "4.19", "4.20"]) == "4.20"


def test_clamp_version_between_available_picks_lower() -> None:
    """Version between available picks highest available <= requested."""
    assert SolrHybridSearch._clamp_version("4.19", ["4.18", "4.20", "4.22"]) == "4.18"


def test_clamp_version_unparseable_returns_as_is() -> None:
    """Unparseable version string is returned unchanged."""
    assert SolrHybridSearch._clamp_version("not-a-version", ["4.18"]) == "not-a-version"


def test_resolve_raises_when_env_not_set() -> None:
    """Missing OCP_CLUSTER_VERSION raises InvalidConfigurationError."""
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(InvalidConfigurationError),
    ):
        SolrHybridSearch._resolve_chunk_filter_query("http://solr:8080", 10.0)


def test_resolve_builds_filter_with_clamped_version() -> None:
    """Env var + Solr facet response produces a version-specific filter query."""
    facet_response = httpx.Response(
        200,
        json={
            "facet_counts": {
                "facet_fields": {"product_version": ["4.18", 10, "4.19", 5, "4.20", 3]}
            }
        },
        request=httpx.Request("GET", "http://solr:8080/solr/portal-rag/select"),
    )
    with (
        patch.dict("os.environ", {"OCP_CLUSTER_VERSION": "4.19.7"}),
        patch("ols.src.rag_index.solr_support.httpx.get", return_value=facet_response),
    ):
        result = SolrHybridSearch._resolve_chunk_filter_query("http://solr:8080", 10.0)
    assert "product_version:4.19" in result
    assert "openshift_container_platform" in result
