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
import os
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
from langchain_core.tools.structured import StructuredTool
from packaging.version import InvalidVersion, Version

from ols.app.models.models import RagChunk
from ols.src.rag.stop_words import ENGLISH_STOP_WORDS
from ols.utils.checks import InvalidConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from ols.app.models.config import SolrHybridSettings

logger = logging.getLogger(__name__)

_ACCESS_BASE = "https://access.redhat.com"

_SOLR_COLLECTION = "portal-rag"
_SOLR_VECTOR_FIELD = "chunk_vector"
_SOLR_CHUNK_TEXT_FIELD = "chunk"

_RAG_CHUNK_EDISMAX_QF = "chunk^3 title^4 headings^2 resourceName^1 product^0.5"
_RAG_CHUNK_EDISMAX_MM = "2<-1 5<75% 10<50%"

_RAG_CHUNK_LEXICAL_HEADROOM_MULT = 5
_RAG_CHUNK_LEXICAL_MIN_ROWS = 15
_RAG_CHUNK_LEXICAL_MAX_ROWS = 80

_MAX_FAMILY_CHUNKS = 50

_TERM_TRIM_CHARS = "?.,!"
_IP_CIDR_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?$")
_NIC_NAME_RE = re.compile(r"^(?:ens|enp|eth|em)\d", re.IGNORECASE)


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


def normalize_solr_hybrid_query(query: str) -> str:
    """Normalize user text for Solr hybrid lexical and embedding paths.

    Strip stopwords and quote hyphenated compounds for stable Solr matching.
    """
    tokens = _split_quoted_and_plain(query)
    parts: list[str] = []
    for t in tokens:
        if t.startswith('"'):
            parts.append(t)
            continue
        stripped = t.rstrip(_TERM_TRIM_CHARS)
        if not stripped:
            continue
        if _IP_CIDR_RE.match(stripped) or _NIC_NAME_RE.match(stripped):
            continue
        norm = stripped.lower().rstrip(_TERM_TRIM_CHARS)
        if re.fullmatch(r"\d+(?:\.\d+)*", norm) or norm not in ENGLISH_STOP_WORDS:
            parts.append(stripped)
    # Quote hyphenated compounds so Solr treats them as phrases, not subtraction
    parts = [
        f'"{t}"' if "-" in t and not t.startswith('"') and len(t) > 3 else t
        for t in parts
    ]
    return " ".join(parts) if parts else query


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


class SolrHybridSearch:
    """Solr hybrid retrieval via ``POST …/hybrid-search`` (lexical-primary + KNN rerank)."""

    def __init__(
        self,
        settings: SolrHybridSettings,
        encode_fn: Callable[[str], Any],
    ) -> None:
        """Store Solr HTTP settings and the embedding function used for ``rqq`` KNN.

        Resolves the OCP product version at construction time by reading the
        ``OCP_CLUSTER_VERSION`` environment variable and querying Solr for
        available versions.  The resolved version is used to build
        ``chunk_filter_query``.

        Args:
            settings: Solr base URL, hybrid weights, timeouts, and row limits.
            encode_fn: Maps query text to a dense vector (e.g. LlamaIndex
                ``HuggingFaceEmbedding.get_text_embedding``).

        Raises:
            InvalidConfigurationError: If ``OCP_CLUSTER_VERSION`` is not set.
        """
        self._settings = settings
        self._encode_fn = encode_fn
        self.chunk_filter_query: str = self._resolve_chunk_filter_query(
            settings.solr_http_base, settings.hybrid_solr_timeout_s
        )

    async def search(self, query: str, token_budget: int = 0) -> list[RetrievedChunk]:
        """Run hybrid-search; return expanded passages capped at ``max_results``, or ``[]``.

        Args:
            query: User query text.
            token_budget: Remaining token budget for tool output. Controls how
                much chunk expansion is performed per matched chunk. When 0,
                chunks are returned without expansion.

        On embedding failures, HTTP errors, JSON decode errors, or malformed Solr payloads,
        logs and returns an empty list so callers can continue without passages.
        """
        try:
            return await self._search_impl(query, token_budget)
        except Exception:
            logger.exception(
                "Solr hybrid search failed for query: %.200s",
                query,
            )
            return []

    async def _search_impl(self, query: str, token_budget: int) -> list[RetrievedChunk]:
        """Execute the hybrid search, dedupe, expand chunks, and build results."""
        cfg = self._settings
        cleaned = normalize_solr_hybrid_query(query)
        base = cfg.solr_http_base.rstrip("/")
        hybrid_url = f"{base}/solr/{_SOLR_COLLECTION}/hybrid-search"

        def _encode() -> list[float]:
            vec = self._encode_fn(cleaned)
            return [float(x) for x in (vec.tolist() if hasattr(vec, "tolist") else vec)]

        loop = asyncio.get_running_loop()
        # Use run_in_executor instead of asyncio.to_thread: to_thread copies
        # contextvars (including OpenTelemetry span tokens) into the worker,
        # which corrupts the OTEL context on detach and can crash the httpx
        # connection pool with "Event loop is closed".
        query_embedding = await loop.run_in_executor(None, _encode)
        vector_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        form = self._build_hybrid_form(cleaned=cleaned, vector_str=vector_str)
        async with httpx.AsyncClient(timeout=cfg.hybrid_solr_timeout_s) as client:
            response = await client.post(
                hybrid_url,
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            payload = _solr_response_json(response, log_url=hybrid_url)
            hybrid_docs = list(payload.get("response", {}).get("docs", []))

            if not hybrid_docs:
                logger.warning("No results (hybrid-search) for: %s", query)
                return []

            if cfg.hybrid_score_threshold > 0:
                hybrid_docs = [
                    d
                    for d in hybrid_docs
                    if _safe_solr_score(d.get("score", 0.0))
                    >= cfg.hybrid_score_threshold
                ]
                if not hybrid_docs:
                    logger.warning(
                        "No docs above hybrid_score_threshold=%s for: %s",
                        cfg.hybrid_score_threshold,
                        query,
                    )
                    return []

            deduped = self._dedupe_by_parent(hybrid_docs)[: cfg.max_results]

            per_chunk_budget = token_budget // len(deduped) if token_budget > 0 else 0

            expanded: list[RetrievedChunk] = []
            for doc in deduped:
                if per_chunk_budget > 0 and cfg.max_expansion_neighbors > 0:
                    family = await self._fetch_family(client, base, doc)
                    ordered = self._expand_around_match(
                        family,
                        doc.get("chunk_index", -1),
                        per_chunk_budget,
                        max_neighbors=cfg.max_expansion_neighbors,
                    )
                else:
                    ordered = [doc]
                chunk = self._assemble_chunk(doc, ordered)
                logger.debug(
                    "Chunk %s: expanded %d→%d siblings (family=%d, budget=%d)",
                    doc.get("chunk_index", "?"),
                    1,
                    len(ordered),
                    (
                        len(family)
                        if per_chunk_budget > 0 and cfg.max_expansion_neighbors > 0
                        else 0
                    ),
                    per_chunk_budget,
                )
                expanded.append(chunk)
            return expanded

    # ------------------------------------------------------------------
    # Startup: OCP version resolution and chunk_filter_query
    # ------------------------------------------------------------------

    _OCP_CLUSTER_VERSION_ENV = "OCP_CLUSTER_VERSION"
    _OCP_PRODUCT = "openshift_container_platform"
    _ROSA_PRODUCT_ENV = "OLS_ROSA_PRODUCT"

    @staticmethod
    def _resolve_chunk_filter_query(solr_http_base: str, timeout_s: float) -> str:
        """Build ``chunk_filter_query`` from the cluster's OCP version.

        When ``OLS_ROSA_PRODUCT`` is set (by the operator on ROSA clusters),
        the filter becomes a compound OR including both OCP and ROSA product
        documentation.

        Raises:
            InvalidConfigurationError: If ``OCP_CLUSTER_VERSION`` is not set
                or Solr is unreachable.
        """
        env_version = os.environ.get(SolrHybridSearch._OCP_CLUSTER_VERSION_ENV)
        if not env_version:
            raise InvalidConfigurationError(
                f"{SolrHybridSearch._OCP_CLUSTER_VERSION_ENV} environment variable "
                "must be set when solr_hybrid is configured"
            )

        env_version = env_version.strip()
        ocp_resolved = SolrHybridSearch._resolve_product_version(
            SolrHybridSearch._OCP_PRODUCT, solr_http_base, timeout_s, env_version
        )

        ocp_filter = (
            f"(product:{SolrHybridSearch._OCP_PRODUCT}"
            f" AND product_version:{ocp_resolved})"
        )

        rosa_product = os.environ.get(SolrHybridSearch._ROSA_PRODUCT_ENV, "").strip()
        if rosa_product:
            try:
                rosa_resolved = SolrHybridSearch._resolve_product_version(
                    rosa_product, solr_http_base, timeout_s, env_version
                )
            except InvalidConfigurationError:
                logger.warning(
                    "ROSA product '%s' not found in Solr — falling back to OCP-only",
                    rosa_product,
                )
            else:
                rosa_filter = (
                    f"(product:{rosa_product} AND product_version:{rosa_resolved})"
                )
                logger.info(
                    "ROSA product detected: product=%s, resolved_version=%s",
                    rosa_product,
                    rosa_resolved,
                )
                return f"is_chunk:true AND ({ocp_filter} OR {rosa_filter})"

        return f"is_chunk:true AND {ocp_filter}"

    @staticmethod
    def _resolve_product_version(
        product: str,
        solr_http_base: str,
        timeout_s: float,
        env_version: str,
    ) -> str:
        """Resolve the best available Solr version for *product*.

        Queries Solr for available versions of the given product, then clamps
        ``env_version`` to the nearest available major.minor.
        """
        available = SolrHybridSearch._fetch_available_product_versions(
            product, solr_http_base, timeout_s
        )
        if not available:
            raise InvalidConfigurationError(
                f"Cannot fetch available versions for product '{product}' "
                f"from Solr at {solr_http_base} — service cannot start"
            )
        resolved = SolrHybridSearch._clamp_version(env_version, available)
        logger.info(
            "Product version resolved: product=%s, env=%s, available=%s, resolved=%s",
            product,
            env_version,
            available,
            resolved,
        )
        return resolved

    _SOLR_STARTUP_RETRIES = 24
    _SOLR_STARTUP_BACKOFF_S = 5

    @staticmethod
    def _fetch_available_product_versions(
        product: str, solr_http_base: str, timeout_s: float
    ) -> list[str]:
        """Query Solr for available ``product_version`` values for *product*.

        Retries up to ``_SOLR_STARTUP_RETRIES`` times with
        ``_SOLR_STARTUP_BACKOFF_S`` seconds between attempts to tolerate
        Solr starting up concurrently with OLS.
        """
        base = solr_http_base.rstrip("/")
        select_url = f"{base}/solr/{_SOLR_COLLECTION}/select"
        params = {
            "q": "*:*",
            "fq": f"product:{product}",
            "rows": "0",
            "facet": "true",
            "facet.field": "product_version",
            "facet.mincount": "1",
            "wt": "json",
        }
        last_error: Exception | None = None
        for attempt in range(SolrHybridSearch._SOLR_STARTUP_RETRIES):
            try:
                response = httpx.get(select_url, params=params, timeout=timeout_s)
                response.raise_for_status()
                data = response.json()
                facet_fields = data.get("facet_counts", {}).get("facet_fields", {})
                raw = facet_fields.get("product_version", [])
                return [
                    raw[i] for i in range(0, len(raw), 2) if isinstance(raw[i], str)
                ]
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Solr not ready (attempt %d/%d): %s",
                    attempt + 1,
                    SolrHybridSearch._SOLR_STARTUP_RETRIES,
                    exc,
                )
                time.sleep(SolrHybridSearch._SOLR_STARTUP_BACKOFF_S)
        logger.error(
            "Failed to fetch versions for product '%s' from Solr after all retries: %s",
            product,
            last_error,
        )
        return []

    @staticmethod
    def _to_major_minor(version_str: str) -> Version:
        """Parse a version string and return only major.minor.

        ``4.19.26`` → ``Version("4.19")``, ``4.18`` → ``Version("4.18")``.
        """
        v = Version(version_str)
        return Version(f"{v.major}.{v.minor}")

    @staticmethod
    def _clamp_version(requested: str, available: list[str]) -> str:
        """Clamp *requested* to the nearest available major.minor version.

        The env var may contain a full ``major.minor.patch`` version (e.g.
        ``4.19.26``) while Solr only indexes ``major.minor`` (e.g. ``4.19``).
        Comparison is done on major.minor only.  When the exact minor version
        is not available, the highest available version ≤ requested is chosen.
        """
        try:
            req = SolrHybridSearch._to_major_minor(requested)
        except InvalidVersion:
            logger.warning("Cannot parse OCP_CLUSTER_VERSION '%s'", requested)
            return requested
        parsed: list[tuple[Version, str]] = []
        for v in available:
            try:
                parsed.append((SolrHybridSearch._to_major_minor(v), v))
            except InvalidVersion:
                continue
        if not parsed:
            return requested
        parsed.sort(key=lambda t: t[0])
        lowest, lowest_str = parsed[0]
        highest, highest_str = parsed[-1]
        if req < lowest:
            return lowest_str
        if req > highest:
            return highest_str
        best_str = lowest_str
        for pv, raw in parsed:
            if pv == req:
                return raw
            if pv < req:
                best_str = raw
        return best_str

    # ------------------------------------------------------------------
    # Private helpers called only from _search_impl
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_chunk(
        doc: dict[str, Any], ordered: list[dict[str, Any]]
    ) -> RetrievedChunk:
        """Merge expanded chunk family into a single ``RetrievedChunk``."""
        texts = [
            _solr_chunk_text_for_rag(f, content_field=_SOLR_CHUNK_TEXT_FIELD)
            for f in ordered
        ]
        merged_text = "\n".join(t for t in texts if t)
        raw_score = doc.get("score")
        try:
            score_f = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError):
            score_f = None
        return RetrievedChunk(
            text=merged_text,
            score=score_f,
            metadata={
                "title": str(doc.get("title") or "").strip(),
                "docs_url": _canonical_url_from_solr_doc(doc),
                "index_origin": "solr_hybrid",
                "chunks_expanded": len(ordered),
            },
        )

    def _build_hybrid_form(
        self,
        *,
        cleaned: str,
        vector_str: str,
    ) -> dict[str, str]:
        """POST body for hybrid-search: lexical edismax ``q`` + ``{!rerank}`` with KNN."""
        cfg = self._settings
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
        return {
            "q": q_text,
            "qf": _RAG_CHUNK_EDISMAX_QF,
            "pf": "chunk^4 title^6",
            "pf2": "chunk^2 title^4",
            "pf3": "chunk^1 title^2",
            "mm": _RAG_CHUNK_EDISMAX_MM,
            "hl": "on",
            "hl.fl": _SOLR_CHUNK_TEXT_FIELD,
            "hl.snippets": "4",
            "hl.fragsize": "480",
            "fq": self.chunk_filter_query,
            "defType": "edismax",
            "hl.q": q_text,
            "rq": rq,
            "rqq": rqq,
            "rows": str(lexical_rows),
            "fl": "*,score,originalScore()",
            "wt": "json",
        }

    @staticmethod
    def _dedupe_by_parent(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep the first (highest-scored) raw Solr doc per ``parent_id``."""
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for doc in docs:
            pid = str(doc.get("parent_id") or doc.get("id") or "").strip()
            if not pid:
                out.append(doc)
                continue
            if pid in seen:
                continue
            seen.add(pid)
            out.append(doc)
        return out

    @staticmethod
    async def _fetch_family(
        client: httpx.AsyncClient,
        base_url: str,
        doc: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Fetch sibling chunks that share ``parent_id`` and ``heading_id``.

        Returns family members ordered by ``chunk_index``, or just the original
        doc wrapped in a list when ``heading_id`` is missing (orphan).
        """
        parent_id = doc.get("parent_id")
        heading_id = doc.get("heading_id")
        if not parent_id or not heading_id:
            return [doc]

        select_url = f"{base_url}/solr/{_SOLR_COLLECTION}/select"
        params = {
            "q": "*:*",
            "fq": (
                f"parent_id:{SolrHybridSearch._solr_escape(parent_id)}"
                f" AND heading_id:{SolrHybridSearch._solr_escape(heading_id)}"
                " AND is_chunk:true"
            ),
            "sort": "chunk_index asc",
            "rows": str(_MAX_FAMILY_CHUNKS),
            "fl": f"id,chunk_index,num_tokens,{_SOLR_CHUNK_TEXT_FIELD},title,"
            "parent_id,heading_id,resourceName,score",
            "wt": "json",
        }
        response = await client.get(select_url, params=params)
        response.raise_for_status()
        payload = _solr_response_json(response, log_url=select_url)
        family = list(payload.get("response", {}).get("docs", []))
        return family or [doc]

    @staticmethod
    def _solr_escape(value: str) -> str:
        """Escape special Solr query characters in a field value."""
        special = r'+-&|!(){}[]^"~*?:\/'
        escaped = []
        for ch in value:
            if ch in special:
                escaped.append("\\")
            escaped.append(ch)
        return "".join(escaped)

    @staticmethod
    def _expand_around_match(
        family: list[dict[str, Any]],
        matched_chunk_index: int,
        token_budget: int,
        max_neighbors: int = 2,
    ) -> list[dict[str, Any]]:
        """Expand bidirectionally from matched chunk within a token budget.

        Starts from the matched chunk and alternates between previous and next
        neighbors, accumulating ``num_tokens`` until *token_budget* is exhausted,
        *max_neighbors* is reached on each side, or all family members are
        included.  The result is sorted by ``chunk_index`` for coherent reading
        order.
        """
        match_pos = None
        for i, doc in enumerate(family):
            if doc.get("chunk_index") == matched_chunk_index:
                match_pos = i
                break
        if match_pos is None:
            return family

        budget = token_budget
        selected = [family[match_pos]]
        budget -= family[match_pos].get("num_tokens", 0)

        lo = match_pos - 1
        hi = match_pos + 1
        added_lo = 0
        added_hi = 0
        lo_done = False
        hi_done = False
        while budget > 0 and not (lo_done and hi_done):
            if lo >= 0 and added_lo < max_neighbors and not lo_done:
                cost = family[lo].get("num_tokens", 0)
                if cost <= budget:
                    selected.append(family[lo])
                    budget -= cost
                    lo -= 1
                    added_lo += 1
                else:
                    lo_done = True
            else:
                lo_done = True
            if hi < len(family) and added_hi < max_neighbors and not hi_done:
                cost = family[hi].get("num_tokens", 0)
                if cost <= budget:
                    selected.append(family[hi])
                    budget -= cost
                    hi += 1
                    added_hi += 1
                else:
                    hi_done = True
            else:
                hi_done = True
        selected.sort(key=lambda d: d.get("chunk_index", 0))
        return selected


def get_openshift_docs_tool(client: SolrHybridSearch) -> StructuredTool:
    """Return a LangChain tool that searches product docs via Solr hybrid RAG.

    The returned tool reads ``tools_token_budget`` from its own ``metadata``
    dict (set by the execution framework before each invocation) and passes it
    to :pymethod:`SolrHybridSearch.search` so chunk expansion respects the
    remaining token budget for the round.

    Args:
        client: Configured ``SolrHybridSearch`` (OKP portal-rag query embeddings).

    Returns:
        Async ``StructuredTool`` the model can invoke with a search query string.
    """
    tool_ref: list[StructuredTool] = []

    async def _search_openshift_documentation(
        search_query: str,
    ) -> list[dict[str, str]] | tuple[list[dict[str, str]], dict[str, Any]]:
        """Run hybrid Solr search and return content blocks for the model.

        Returns a list of ``{"text": ...}`` dicts so the tool output framework
        can accumulate blocks until the per-tool token budget is reached.
        On success, returns a tuple whose second element carries
        ``referenced_documents`` for the API response.
        """
        token_budget = (
            (tool_ref[0].metadata or {}).get("tools_token_budget", 0) if tool_ref else 0
        )
        try:
            chunks = await client.search(search_query, token_budget=token_budget)
        except Exception:
            logger.exception("OpenShift docs tool: Solr search failed")
            return [
                {
                    "text": json.dumps(
                        {
                            "error": "documentation_search_failed",
                            "detail": "search_unexpected_error",
                        }
                    )
                }
            ]

        seen: set[str] = set()
        ref_docs: list[RagChunk] = []
        for c in chunks:
            url = c.metadata.get("docs_url", "")
            if url and url not in seen:
                seen.add(url)
                ref_docs.append(
                    RagChunk(
                        text="", doc_url=url, doc_title=c.metadata.get("title", "")
                    )
                )

        results = [
            {
                "text": json.dumps(
                    {
                        "text": c.text,
                        "score": c.score,
                        "title": c.metadata.get("title"),
                        "docs_url": c.metadata.get("docs_url"),
                    },
                    ensure_ascii=False,
                )
            }
            for c in chunks
        ]
        return results, {"referenced_documents": ref_docs}

    tool = StructuredTool.from_function(
        coroutine=_search_openshift_documentation,
        name="search_openshift_documentation",
        description=(
            "Search published Red Hat OpenShift and related product documentation "
            "(not the live cluster). Returns JSON: an array of "
            "{text, score, title, docs_url}, or [] if no hits, or an object with "
            "error on failure. Cite docs_url when using a passage."
        ),
        metadata={"tools_token_budget": 0, "annotations": {"readOnlyHint": True}},
    )
    tool_ref.append(tool)
    return tool
