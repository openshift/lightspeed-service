"""RHOKP MCP client for OpenShift documentation search."""

import asyncio
import json
import logging
import os
import re
from collections.abc import Iterator
from typing import Any, TypedDict

from langchain_core.tools import BaseTool
from langchain_core.tools.structured import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from ols import constants
from ols.app.models.config import OpenShiftDocsMCPSettings
from ols.app.models.models import RagChunk

logger = logging.getLogger(__name__)


class ProductFilter(TypedDict):
    """RHOKP search product filter input."""

    product: str
    versions: list[str]


_VERSION_PATTERN = re.compile(r"\s*(\d+)\.(\d+)(?:\.\d+)?\s*")
_OPENSHIFT_PRODUCT = "openshift_container_platform"
_ROSA_PRODUCT_ENV = "OLS_ROSA_PRODUCT"
_CLUSTER_VERSION_ENV = "OCP_CLUSTER_VERSION"
_RHOKP_TOOL_NAME = "search"
_RHOKP_READINESS_RETRIES = 36
_RHOKP_READINESS_BACKOFF_S = 5
_RHOKP_HTTP_TIMEOUT_S = 60.0
_RHOKP_SEARCH_TOOLS: dict[str, BaseTool] = {}


async def _get_rhokp_search_tool(client: MultiServerMCPClient) -> BaseTool:
    """Retry tool discovery until RHOKP MCP exposes its search tool."""
    last_error: Exception | None = None
    for attempt in range(1, _RHOKP_READINESS_RETRIES + 1):
        try:
            tools = await client.get_tools(server_name="rhokp")
            search_tool = next(
                (tool for tool in tools if tool.name == _RHOKP_TOOL_NAME), None
            )
            if search_tool is None:
                raise RuntimeError("RHOKP MCP search tool is unavailable")
            return search_tool
        except Exception as error:
            last_error = error
            logger.warning(
                "RHOKP MCP not ready (attempt %d/%d): %s",
                attempt,
                _RHOKP_READINESS_RETRIES,
                error,
            )
            if attempt < _RHOKP_READINESS_RETRIES:
                await asyncio.sleep(_RHOKP_READINESS_BACKOFF_S)

    raise RuntimeError(
        f"RHOKP MCP search tool unavailable after {_RHOKP_READINESS_RETRIES} attempts"
    ) from last_error


async def initialize_rhokp_mcp(settings: OpenShiftDocsMCPSettings) -> None:
    """Wait for RHOKP MCP during service startup and cache its search tool."""
    url = str(settings.url)
    if url in _RHOKP_SEARCH_TOOLS:
        return

    client = MultiServerMCPClient(
        {
            "rhokp": {
                "transport": "streamable_http",
                "url": url,
                "timeout": _RHOKP_HTTP_TIMEOUT_S,
            }
        }
    )
    _RHOKP_SEARCH_TOOLS[url] = await _get_rhokp_search_tool(client)
    logger.info("RHOKP MCP search tool is ready")


def _product_filters() -> list[ProductFilter] | None:
    """Build RHOKP product/version filters from the cluster environment."""
    version = os.environ.get(_CLUSTER_VERSION_ENV, "")
    match = _VERSION_PATTERN.fullmatch(version)
    if match is None:
        return None

    major_minor = f"{match.group(1)}.{match.group(2)}"
    products: list[ProductFilter] = [
        {"product": _OPENSHIFT_PRODUCT, "versions": [major_minor]}
    ]
    rosa_product = os.environ.get(_ROSA_PRODUCT_ENV, "").strip()
    if rosa_product:
        products.append({"product": rosa_product, "versions": [major_minor]})
    return products


def _normalize_result_for_model(value: Any) -> Any:
    """Add the documented ``docs_url`` alias to RHOKP citation URLs."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        normalized = _normalize_result_for_model(decoded)
        if normalized != decoded:
            return json.dumps(normalized, ensure_ascii=False, default=str)
        return value
    if isinstance(value, dict):
        normalized = {
            key: _normalize_result_for_model(nested) for key, nested in value.items()
        }
        online_source_url = normalized.get("online_source_url")
        if "docs_url" not in normalized and isinstance(online_source_url, str):
            normalized["docs_url"] = online_source_url
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_result_for_model(item) for item in value]
    return value


def _serialize_result(result: Any) -> str:
    """Serialize normalized MCP search output for the model and result processor."""
    normalized = _normalize_result_for_model(result)
    if isinstance(normalized, str):
        return normalized
    return json.dumps(normalized, ensure_ascii=False, default=str)


def _iter_json_values(value: Any) -> Iterator[dict[str, Any]]:
    """Yield nested JSON values, decoding JSON strings where possible."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return
        yield from _iter_json_values(decoded)
    elif isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_json_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_json_values(nested)


def _referenced_documents(result: Any) -> list[RagChunk]:
    """Extract citation URLs and titles from RHOKP search results."""
    references: list[RagChunk] = []
    seen_urls: set[str] = set()
    for item in _iter_json_values(result):
        url = item.get("online_source_url") or item.get("docs_url")
        if not isinstance(url, str) or not url or url in seen_urls:
            continue
        title = item.get("title") or item.get("doc_title") or ""
        references.append(RagChunk(text="", doc_url=url, doc_title=str(title)))
        seen_urls.add(url)
    return references


def get_openshift_docs_tool(settings: OpenShiftDocsMCPSettings) -> StructuredTool:
    """Create the local docs tool backed by RHOKP's built-in MCP search tool."""
    tool_refs: list[StructuredTool] = []

    async def _search(
        query: str,
    ) -> list[dict[str, str]] | tuple[list[dict[str, str]], dict[str, Any]]:
        """Call RHOKP search with code-enforced cluster filters."""
        products = _product_filters()
        if products is None:
            return [
                {
                    "text": json.dumps(
                        {
                            "error": "documentation_search_failed",
                            "detail": "cluster_version_unavailable",
                        }
                    )
                }
            ]

        search_tool = _RHOKP_SEARCH_TOOLS.get(str(settings.url))
        if search_tool is None:
            return [
                {
                    "text": json.dumps(
                        {
                            "error": "documentation_search_failed",
                            "detail": "mcp_not_initialized",
                        }
                    )
                }
            ]

        try:
            result = await search_tool.ainvoke(
                {
                    "query": query,
                    "products": products,
                    "rows": settings.max_results,
                }
            )
        except Exception:
            logger.exception("OpenShift docs tool: RHOKP MCP search failed")
            return [
                {
                    "text": json.dumps(
                        {
                            "error": "documentation_search_failed",
                            "detail": "mcp_search_failed",
                        }
                    )
                }
            ]

        references = _referenced_documents(result)
        output = [{"text": _serialize_result(result)}]
        return output, {"referenced_documents": references}

    tool = StructuredTool.from_function(
        coroutine=_search,
        name=constants.OPENSHIFT_DOCS_TOOL_NAME,
        description=(
            "Search published Red Hat OpenShift documentation through RHOKP. "
            "Returns documentation results and citations; cite source URLs from results."
        ),
        metadata={
            constants.OLS_INTERNAL_TOOL_METADATA_KEY: constants.OPENSHIFT_DOCS_TOOL_NAME,
            "tools_token_budget": 0,
            "annotations": {"readOnlyHint": True},
        },
    )
    tool_refs.append(tool)
    return tool
