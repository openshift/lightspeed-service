"""Solr hybrid RAG client for OLS (portal-rag / hybrid-search).

Runs ``POST /hybrid-search`` with **lexical-primary** edismax ``q`` and ``{!rerank}``
where ``rqq`` is ``{!knn f=chunk_vector …}[vector]`` (SolrVectorIO / OKP shape). A
KNN-first ``q`` can return empty ``response.docs`` when the vector leg finds no
neighbors even though lexical would match.

Results are **deduped by parent** (first hit per ``parent_id`` / ``id``), then
truncated to ``SolrHybridSettings.max_results``.
"""

from __future__ import annotations

import asyncio
import html as html_mod
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import httpx
from langchain_core.tools.structured import StructuredTool
from llama_index.core.schema import NodeWithScore, TextNode

from ols.constants import RAG_SIMILARITY_CUTOFF
from ols.src.rag.stop_words import ENGLISH_STOP_WORDS

if TYPE_CHECKING:
    from collections.abc import Callable

    from ols.app.models.config import SolrHybridSettings

logger = logging.getLogger(__name__)


def _safe_solr_score(value: Any) -> float:
    """Coerce Solr ``score`` field to float; missing or invalid values become ``0.0``."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _solr_response_json(response: httpx.Response, *, log_url: str) -> dict[str, Any]:
    """Parse Solr JSON body; return empty dict on decode failure or non-object root."""
    try:
        data = response.json()
    except ValueError:
        logger.exception("Solr response JSON decode failed for %s", log_url)
        return {}
    return data if isinstance(data, dict) else {}


_ACCESS_BASE = "https://access.redhat.com"

_SOLR_COLLECTION = "portal-rag"
_SOLR_VECTOR_FIELD = "chunk_vector"
_SOLR_CHUNK_TEXT_FIELD = "chunk"

_RAG_CHUNK_EDISMAX_QF = "chunk^3 title^4 headings^2 resourceName^1 product^0.5"
_RAG_CHUNK_EDISMAX_MM = "2<-1 5<75% 10<50%"

_RAG_CHUNK_LEXICAL_HEADROOM_MULT = 5
_RAG_CHUNK_LEXICAL_MIN_ROWS = 15
_RAG_CHUNK_LEXICAL_MAX_ROWS = 80

_TERM_TRIM_CHARS = "?.,!"

# Skip tokens that are unlikely to be useful Solr keywords (noise in support tickets / logs).
_IP_CIDR_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?$")
_NIC_NAME_RE = re.compile(r"^(?:ens|enp|eth|em)\d", re.IGNORECASE)


def _split_quoted_and_plain(text: str) -> list[str]:
    """Split text on whitespace while keeping double-quoted spans as single tokens.

    Args:
        text: Raw user input.

    Returns:
        Ordered tokens; quoted phrases appear as ``"..."`` elements.
    """
    tokens: list[str] = []
    remainder = text
    while '"' in remainder:
        before, _, rest = remainder.partition('"')
        tokens.extend(before.split())
        if '"' not in rest:
            tokens.extend(rest.split())
            remainder = ""
            break
        phrase, _, remainder = rest.partition('"')
        if phrase:
            tokens.append(f'"{phrase}"')
    tokens.extend(remainder.split())
    return tokens


def _is_numeric(token: str) -> bool:
    """Return whether ``token`` is a version-style number after trimming trailing punctuation.

    Args:
        token: Single token from the split query.

    Returns:
        True when the token matches only digits and dot-separated digit groups.
    """
    normalized = token.lower().rstrip(_TERM_TRIM_CHARS)
    return bool(re.fullmatch(r"\d+(?:\.\d+)*", normalized))


def _is_network_noise(token: str) -> bool:
    """Return whether ``token`` is treated as non-query noise (IP/CIDR or NIC-like name).

    Args:
        token: Single token from the split query.

    Returns:
        True when the token matches IPv4/CIDR or common Linux interface name patterns.
    """
    stripped = token.rstrip(_TERM_TRIM_CHARS)
    return bool(_IP_CIDR_RE.match(stripped) or _NIC_NAME_RE.match(stripped))


def normalize_solr_hybrid_query(query: str) -> str:
    """Normalize user text for Solr hybrid lexical and embedding paths.

    Strip stopwords and quote hyphenated compounds for stable Solr matching.
    """
    # Split into quoted spans (preserved verbatim) and plain tokens
    tokens = _split_quoted_and_plain(query)
    parts: list[str] = []
    for t in tokens:
        if t.startswith('"'):
            parts.append(t)
            continue
        # Strip trailing punctuation and drop empty / noise tokens
        stripped = t.rstrip(_TERM_TRIM_CHARS)
        if not stripped:
            continue
        # Drop IP/CIDR addresses and NIC-like names (e.g. eth0, ens3)
        if _is_network_noise(stripped):
            continue
        # Keep version-style numbers; drop English stop words
        if _is_numeric(stripped) or stripped.lower() not in ENGLISH_STOP_WORDS:
            parts.append(stripped)
    # Quote hyphenated compounds so Solr treats them as phrases, not subtraction
    parts = [
        f'"{t}"' if "-" in t and not t.startswith('"') and len(t) > 3 else t
        for t in parts
    ]
    return " ".join(parts) if parts else query


def _rag_chunk_edismax_core_params(
    *,
    cleaned_query: str,
    chunk_filter_query: str | None,
) -> dict[str, str]:
    """Edismax main-query fields shared by chunk ``/select`` and hybrid ``POST``."""
    params: dict[str, str] = {
        "q": cleaned_query,
        "qf": _RAG_CHUNK_EDISMAX_QF,
        "pf": "chunk^4 title^6",
        "pf2": "chunk^2 title^4",
        "pf3": "chunk^1 title^2",
        "mm": _RAG_CHUNK_EDISMAX_MM,
        "hl": "on",
        "hl.fl": _SOLR_CHUNK_TEXT_FIELD,
        "hl.snippets": "4",
        "hl.fragsize": "480",
    }
    if chunk_filter_query:
        params["fq"] = chunk_filter_query
    return params


def _canonical_url_from_solr_doc(doc: dict[str, Any]) -> str:
    """Resolve a citation URL from portal-rag chunk field values.

    Args:
        doc: One Solr document dict.

    Returns:
        Absolute ``https://`` URL when possible; empty string if none found.
    """
    for key in ("resourceName", "view_uri"):
        v = doc.get(key)
        if v:
            s = str(v).strip()
            if s.startswith("http"):
                return s
            if s.startswith("/"):
                return f"{_ACCESS_BASE}{s}"
    vid = doc.get("id")
    if vid:
        s = str(vid).strip()
        if s.startswith("http"):
            return s
        if s.startswith("/"):
            return f"{_ACCESS_BASE}{s}"
    return ""


def _strip_html(text: str) -> str:
    """Turn HTML-ish chunk text into plain space-normalized text for the LLM.

    Args:
        text: Raw string that may contain entities and tags.

    Returns:
        Decoded, tag-stripped, whitespace-collapsed text.
    """
    t = html_mod.unescape(text)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _solr_chunk_text_for_rag(doc: dict[str, Any], *, content_field: str) -> str:
    """Extract chunk body text from a Solr document and strip HTML for RAG.

    Args:
        doc: One Solr document dict.
        content_field: Primary field name for chunk text (e.g. ``chunk``).

    Returns:
        Plain text suitable for prompt context, or empty string if missing.
    """
    for key in (content_field, "chunk", "main_content"):
        raw = doc.get(key)
        if raw:
            return _strip_html(str(raw).strip())
    return ""


@dataclass(frozen=True)
class RetrievedChunk:
    """One ranked passage from Solr for downstream RAG."""

    text: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _parent_dedupe_key(chunk: RetrievedChunk) -> str | None:
    meta = chunk.metadata
    pid = meta.get("parent_id")
    if pid is not None and str(pid).strip():
        return str(pid).strip()
    did = meta.get("id")
    if did is not None and str(did).strip():
        return str(did).strip()
    return None


def dedupe_retrieved_chunks_by_parent(
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Keep the first chunk per Solr parent book (``parent_id`` or ``id``)."""
    if not chunks:
        return []
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for c in chunks:
        key = _parent_dedupe_key(c)
        if key is None:
            out.append(c)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _retrieved_chunks_from_solr_docs(
    docs: list[dict[str, Any]],
    *,
    content_field: str,
    passage_origin: str = "solr_hybrid",
) -> list[RetrievedChunk]:
    """Map Solr ``response.docs`` rows to ``RetrievedChunk`` for prompt and citations.

    Args:
        docs: Ordered hit list from a Solr JSON response.
        content_field: Primary body field on each doc (e.g. ``chunk``).
        passage_origin: ``index_origin`` value stored on each chunk (e.g. ``solr_hybrid``).

    Returns:
        One ``RetrievedChunk`` per doc with text, optional score, and metadata.
    """
    out: list[RetrievedChunk] = []
    for doc in docs:
        text = _solr_chunk_text_for_rag(doc, content_field=content_field).strip()
        title = str(doc.get("title") or doc.get("allTitle") or "").strip()
        raw_score = doc.get("score")
        try:
            score_f = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError):
            score_f = None
        url = _canonical_url_from_solr_doc(doc)
        meta: dict[str, Any] = {
            "title": title,
            "docs_url": url,
            "index_origin": passage_origin,
        }
        for k in ("resourceName", "view_uri", "id", "parent_id", "allTitle"):
            v = doc.get(k)
            if v is not None and str(v).strip() != "":
                meta[k] = v
        out.append(RetrievedChunk(text=text, score=score_f, metadata=meta))
    return out


def nodes_from_solr_retrieved_chunks(
    chunks: list[RetrievedChunk],
) -> list[NodeWithScore]:
    """Map Solr passages to LlamaIndex nodes for shared RAG truncation and budgeting.

    Args:
        chunks: Ordered Solr hit list from :meth:`SolrHybridSearch.search`.

    Returns:
        ``NodeWithScore`` list suitable for :meth:`TokenHandler.truncate_rag_context`.
    """
    nodes: list[NodeWithScore] = []
    for rank, chunk in enumerate(chunks):
        body = chunk.text.strip()
        if not body:
            continue
        meta = dict(chunk.metadata)
        meta.setdefault("index_origin", "solr_hybrid")
        meta.setdefault("index_id", "")
        raw = chunk.score
        if raw is None:
            score = 1.0 - rank * 1e-6
        else:
            score = float(raw)
            if score < RAG_SIMILARITY_CUTOFF:
                score = RAG_SIMILARITY_CUTOFF + 0.01 / (rank + 1)
        node = TextNode(text=body, metadata=meta)
        nodes.append(NodeWithScore(node=node, score=score))
    return nodes


def _build_lexical_first_hybrid_form(
    cfg: SolrHybridSettings,
    *,
    cleaned: str,
    vector_str: str,
) -> dict[str, str]:
    """POST body for hybrid-search: lexical edismax ``q`` + ``{!rerank}`` with ``rqq`` = ``{!knn}``.

    Matches SolrVectorIO (lexical primary ``q``; KNN-only main ``q`` can yield empty
    ``response.docs`` when the vector leg has no neighbors despite lexical matches).
    """
    lexical_fetch_rows = min(
        max(
            cfg.max_results * _RAG_CHUNK_LEXICAL_HEADROOM_MULT,
            _RAG_CHUNK_LEXICAL_MIN_ROWS,
        ),
        _RAG_CHUNK_LEXICAL_MAX_ROWS,
    )
    lexical_rows = max(cfg.hybrid_pool_docs, lexical_fetch_rows)
    q_text = cleaned.replace("?", "").replace("*", "")
    rq = (
        f"{{!rerank reRankQuery=$rqq reRankDocs={cfg.hybrid_pool_docs} "
        f"reRankWeight={cfg.hybrid_vector_boost}}}"
    )
    rqq = f"{{!knn f={_SOLR_VECTOR_FIELD} topK={lexical_rows}}}{vector_str}"
    form = _rag_chunk_edismax_core_params(
        cleaned_query=q_text,
        chunk_filter_query=cfg.chunk_filter_query,
    )
    form["defType"] = "edismax"
    form["hl.q"] = q_text
    form["rq"] = rq
    form["rqq"] = rqq
    form["rows"] = str(lexical_rows)
    form["fl"] = "*,score,originalScore()"
    form["wt"] = "json"
    return form


async def _post_hybrid(
    url: str,
    form: dict[str, str],
    client_timeout_s: float,
) -> dict[str, Any]:
    """POST form data to Solr and return the parsed JSON response.

    Args:
        url: Full hybrid-search endpoint URL.
        form: Flat form key/value pairs.
        client_timeout_s: Total HTTP timeout in seconds.

    Returns:
        Parsed JSON object (typically includes ``response.docs``). On invalid JSON
        body after a successful status, logs and returns an empty dict.

    Raises:
        httpx.HTTPError: When the response status is not successful.
    """
    async with httpx.AsyncClient(timeout=client_timeout_s) as client:
        response = await client.post(
            url,
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        return _solr_response_json(response, log_url=url)


def _retrieved_chunks_from_hybrid_raw_docs(
    docs: list[dict[str, Any]],
    *,
    hybrid_score_threshold: float,
    content_field: str,
) -> list[RetrievedChunk]:
    """Turn hybrid ``response.docs`` into passages, applying optional score cutoff."""
    if not docs:
        return []
    if hybrid_score_threshold <= 0:
        return _retrieved_chunks_from_solr_docs(
            docs,
            content_field=content_field,
            passage_origin="solr_hybrid",
        )
    filtered_docs: list[dict[str, Any]] = []
    for doc in docs:
        score = _safe_solr_score(doc.get("score", 0.0))
        if score < hybrid_score_threshold:
            continue
        filtered_docs.append(doc)
    if not filtered_docs:
        return []
    return _retrieved_chunks_from_solr_docs(
        filtered_docs,
        content_field=content_field,
        passage_origin="solr_hybrid",
    )


class SolrHybridSearch:
    """Solr hybrid retrieval via ``POST …/hybrid-search`` (lexical-primary + KNN rerank)."""

    def __init__(
        self,
        settings: SolrHybridSettings,
        encode_fn: Callable[[str], Any],
    ) -> None:
        """Store Solr HTTP settings and the embedding function used for ``rqq`` KNN.

        Args:
            settings: Solr base URL, hybrid weights, timeouts, and row limits.
            encode_fn: Maps query text to a dense vector (e.g. LlamaIndex
                ``HuggingFaceEmbedding.get_text_embedding``).
        """
        self._settings = settings
        self._encode_fn = encode_fn

    async def search(self, query: str) -> list[RetrievedChunk]:
        """Run hybrid-search; return deduped passages capped at ``max_results``, or ``[]``.

        On embedding failures, HTTP errors, JSON decode errors, or malformed Solr payloads,
        logs and returns an empty list so callers can continue without passages.
        """
        try:
            return await self._search_impl(query)
        except Exception:
            logger.exception(
                "Solr hybrid search failed for query: %.200s",
                query,
            )
            return []

    async def _search_impl(self, query: str) -> list[RetrievedChunk]:
        cfg = self._settings
        cleaned = normalize_solr_hybrid_query(query)
        base = cfg.solr_http_base.rstrip("/")
        hybrid_url = f"{base}/solr/{_SOLR_COLLECTION}/hybrid-search"

        def _encode() -> list[float]:
            vec = self._encode_fn(cleaned)
            return [float(x) for x in (vec.tolist() if hasattr(vec, "tolist") else vec)]

        query_embedding = await asyncio.to_thread(_encode)
        vector_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        form = _build_lexical_first_hybrid_form(
            cfg, cleaned=cleaned, vector_str=vector_str
        )
        payload = await _post_hybrid(hybrid_url, form, cfg.hybrid_solr_timeout_s)
        hybrid_docs = list(payload.get("response", {}).get("docs", []))

        hybrid_chunks = _retrieved_chunks_from_hybrid_raw_docs(
            hybrid_docs,
            hybrid_score_threshold=cfg.hybrid_score_threshold,
            content_field=_SOLR_CHUNK_TEXT_FIELD,
        )
        if hybrid_chunks:
            return dedupe_retrieved_chunks_by_parent(hybrid_chunks)[: cfg.max_results]

        if not hybrid_docs:
            logger.warning("No results (hybrid-search) for: %s", query)
        elif cfg.hybrid_score_threshold > 0:
            logger.warning(
                "No docs above hybrid_score_threshold=%s for: %s",
                cfg.hybrid_score_threshold,
                query,
            )
        return []


def solr_hybrid_openshift_docs_tool_uses_client_lookup(
    solr_hybrid: SolrHybridSettings | None,
) -> bool:
    """Return True when Solr hybrid is configured and docs use the tool path."""
    return solr_hybrid is not None and not solr_hybrid.solr_direct_rag


def solr_hybrid_openshift_docs_tool_active(
    solr_hybrid: SolrHybridSettings | None,
    solr_client: SolrHybridSearch | None,
) -> bool:
    """Return True for tool-only Solr docs when hybrid is enabled and a client exists."""
    uses = solr_hybrid_openshift_docs_tool_uses_client_lookup(solr_hybrid)
    return uses and solr_client is not None


def get_openshift_docs_tool(client: SolrHybridSearch) -> StructuredTool:
    """Return a LangChain tool that searches product docs via Solr hybrid RAG.

    Args:
        client: Configured ``SolrHybridSearch`` (OKP portal-rag query embeddings).

    Returns:
        Async ``StructuredTool`` the model can invoke with a search query string.
    """

    async def _search_openshift_documentation(search_query: str) -> str:
        """Run hybrid Solr search and return JSON passages for the model."""
        try:
            chunks = await client.search(search_query)
        except Exception:
            logger.exception("OpenShift docs tool: Solr search failed")
            return json.dumps(
                {
                    "error": "documentation_search_failed",
                    "detail": "search_unexpected_error",
                }
            )
        rows = [
            {
                "text": c.text,
                "score": c.score,
                "title": c.metadata.get("title"),
                "docs_url": c.metadata.get("docs_url"),
            }
            for c in chunks
        ]
        return json.dumps(rows, ensure_ascii=False)

    return StructuredTool.from_function(
        coroutine=_search_openshift_documentation,
        name="search_openshift_documentation",
        description=(
            "Search published Red Hat OpenShift and related product documentation "
            "(not the live cluster). Returns JSON: an array of "
            "{text, score, title, docs_url}, or [] if no hits, or an object with "
            "error on failure. Cite docs_url when using a passage."
        ),
    )


def append_openshift_docs_tool_if_configured(
    tools: list[StructuredTool],
    *,
    solr_hybrid: SolrHybridSettings | None,
    solr_client: SolrHybridSearch | None,
) -> None:
    """Append ``search_openshift_documentation`` for tool-only Solr (no direct RAG merge)."""
    if not solr_hybrid_openshift_docs_tool_active(solr_hybrid, solr_client):
        return
    tools.append(get_openshift_docs_tool(cast("SolrHybridSearch", solr_client)))
