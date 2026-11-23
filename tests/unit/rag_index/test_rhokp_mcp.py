"""Tests for the RHOKP MCP documentation search tool."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ols import constants
from ols.app.models.config import OpenShiftDocsMCPSettings
from ols.app.models.models import RagChunk
from ols.src.rag_index import rhokp_mcp
from ols.src.rag_index.rhokp_mcp import (
    get_openshift_docs_tool,
    initialize_rhokp_mcp,
)


@pytest.fixture(autouse=True)
def clear_search_tool_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear cached RHOKP tools between tests."""
    monkeypatch.setattr(rhokp_mcp, "_RHOKP_SEARCH_TOOLS", {})


@pytest.mark.asyncio
async def test_docs_tool_calls_rhokp_search_with_enforced_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call RHOKP's search tool with cluster-derived filters and preserve citations."""
    monkeypatch.setenv("OCP_CLUSTER_VERSION", "4.19.0")
    monkeypatch.setenv("OLS_ROSA_PRODUCT", "rosa_product")
    response = {
        "results": [
            {
                "title": "Virtualization",
                "text": "Live migration documentation.",
                "online_source_url": "https://docs.redhat.com/virtualization",
            }
        ]
    }
    remote_tool = MagicMock()
    remote_tool.name = "search"
    remote_tool.ainvoke = AsyncMock(return_value=response)
    mcp_client = MagicMock()
    mcp_client.get_tools = AsyncMock(return_value=[remote_tool])

    with patch(
        "ols.src.rag_index.rhokp_mcp.MultiServerMCPClient",
        return_value=mcp_client,
    ) as mcp_client_class:
        settings = OpenShiftDocsMCPSettings(
            url="https://lightspeed-rhokp.test.svc:8443/mcp",
            max_results=12,
        )
        await initialize_rhokp_mcp(settings)
        tool = get_openshift_docs_tool(settings)
        result = await tool.ainvoke({"query": "live migration"})

    mcp_client_class.assert_called_once_with(
        {
            "rhokp": {
                "transport": "streamable_http",
                "url": "https://lightspeed-rhokp.test.svc:8443/mcp",
                "timeout": 60.0,
            }
        }
    )
    mcp_client.get_tools.assert_awaited_once_with(server_name="rhokp")
    remote_tool.ainvoke.assert_awaited_once_with(
        {
            "query": "live migration",
            "products": [
                {"product": "openshift_container_platform", "versions": ["4.19"]},
                {"product": "rosa_product", "versions": ["4.19"]},
            ],
            "rows": 12,
        }
    )
    expected_model_response = {
        "results": [
            {
                **response["results"][0],
                "docs_url": response["results"][0]["online_source_url"],
            }
        ]
    }
    assert result == (
        [{"text": json.dumps(expected_model_response, ensure_ascii=False)}],
        {
            "referenced_documents": [
                RagChunk(
                    text="",
                    doc_url="https://docs.redhat.com/virtualization",
                    doc_title="Virtualization",
                )
            ]
        },
    )
    assert tool.name == "search"
    assert tool.metadata[constants.OLS_INTERNAL_TOOL_METADATA_KEY] == "search"
    assert "mcp_server" not in tool.metadata


@pytest.mark.asyncio
async def test_docs_tool_retries_until_rhokp_search_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry RHOKP tool discovery until the MCP server exposes ``search``."""
    monkeypatch.setenv("OCP_CLUSTER_VERSION", "4.19.0")
    remote_tool = MagicMock()
    remote_tool.name = "search"
    remote_tool.ainvoke = AsyncMock(return_value={"results": []})
    mcp_client = MagicMock()
    mcp_client.get_tools = AsyncMock(
        side_effect=[ConnectionError("not ready"), [], [remote_tool]]
    )

    with (
        patch(
            "ols.src.rag_index.rhokp_mcp.MultiServerMCPClient",
            return_value=mcp_client,
        ),
        patch("asyncio.sleep", new_callable=AsyncMock) as sleep,
        patch("ols.src.rag_index.rhokp_mcp._RHOKP_READINESS_RETRIES", 3),
        patch(
            "ols.src.rag_index.rhokp_mcp._RHOKP_READINESS_BACKOFF_S",
            0.25,
        ),
    ):
        settings = OpenShiftDocsMCPSettings(
            url="https://lightspeed-rhokp.test.svc:8443/mcp"
        )
        await initialize_rhokp_mcp(settings)
        await initialize_rhokp_mcp(settings)
        tool = get_openshift_docs_tool(settings)
        await tool.ainvoke({"query": "live migration"})

    assert mcp_client.get_tools.await_count == 3
    assert sleep.await_args_list[0].args == (0.25,)
    assert sleep.await_args_list[1].args == (0.25,)
    remote_tool.ainvoke.assert_awaited_once()


@pytest.mark.parametrize("version", ["unknown", "4.19invalid", "4.19.0.1"])
@pytest.mark.asyncio
async def test_docs_tool_fails_closed_for_invalid_cluster_version(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    """Do not call RHOKP when the cluster version is unavailable."""
    monkeypatch.setenv("OCP_CLUSTER_VERSION", version)

    with patch("ols.src.rag_index.rhokp_mcp.MultiServerMCPClient") as mcp_client_class:
        tool = get_openshift_docs_tool(
            OpenShiftDocsMCPSettings(url="https://lightspeed-rhokp.test.svc:8443/mcp")
        )
        result = await tool.ainvoke({"query": "live migration"})

    mcp_client_class.assert_not_called()
    assert result == [
        {
            "text": json.dumps(
                {
                    "error": "documentation_search_failed",
                    "detail": "cluster_version_unavailable",
                }
            )
        }
    ]
