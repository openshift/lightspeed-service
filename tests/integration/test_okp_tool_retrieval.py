"""Integration tests for OKP documentation tool retrieval through query endpoints."""

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from ols import config, constants
from ols.app.models.config import OpenShiftDocsMCPSettings
from ols.src.rag_index import rhokp_mcp
from ols.src.rag_index.rhokp_mcp import initialize_rhokp_mcp
from ols.utils import suid
from tests.mock_classes.mock_langchain_interface import mock_langchain_interface
from tests.mock_classes.mock_llm_loader import mock_llm_loader


@pytest.fixture
def client() -> TestClient:
    """Create an API client with the integration configuration."""
    config.reload_from_yaml_file("tests/config/config_for_integration_tests.yaml")
    from ols.app.main import app  # pylint: disable=import-outside-toplevel

    return TestClient(app)


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_okp_tool_retrieval_reaches_referenced_documents(
    client: TestClient, endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose citations from a forced OKP tool call on both query endpoints."""
    url = "https://docs.redhat.com/en/documentation/openshift_container_platform/4.19/html/virtualization"
    settings = OpenShiftDocsMCPSettings(
        url="https://lightspeed-rhokp.test.svc:8443/mcp"
    )
    config.ols_config.openshift_docs_mcp = settings
    monkeypatch.setattr(rhokp_mcp, "_RHOKP_SEARCH_TOOLS", {})
    monkeypatch.setenv("OCP_CLUSTER_VERSION", "4.19.0")
    monkeypatch.delenv("OLS_ROSA_PRODUCT", raising=False)
    remote_tool = MagicMock()
    remote_tool.name = "search"
    remote_tool.ainvoke = AsyncMock(
        return_value={
            "results": [
                {
                    "title": "OpenShift Virtualization",
                    "text": "OpenShift Virtualization runs virtual machines.",
                    "online_source_url": url,
                }
            ]
        }
    )
    mcp_client = MagicMock()
    mcp_client.get_tools = AsyncMock(return_value=[remote_tool])
    config.ols_config.guardrails.tool_result_inspection.enabled = False

    rounds = 0

    async def model_response(
        *args: object, **kwargs: object
    ) -> AsyncIterator[AIMessageChunk]:
        nonlocal rounds
        rounds += 1
        if rounds == 1:
            yield AIMessageChunk(
                content="",
                response_metadata={"finish_reason": "tool_calls"},
                tool_call_chunks=[
                    {
                        "name": "search",
                        "args": '{"query": "openshift virtualization"}',
                        "id": "okp-call-1",
                        "index": 0,
                    }
                ],
            )
        else:
            yield AIMessageChunk(content="OpenShift Virtualization runs VMs.")
            yield AIMessageChunk(
                content="", response_metadata={"finish_reason": "stop"}
            )

    with (
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "ols.src.rag_index.rhokp_mcp.MultiServerMCPClient",
            return_value=mcp_client,
        ),
        patch("ols.src.query_helpers.query_helper.load_llm", new=mock_llm_loader(None)),
        patch(
            "ols.src.query_helpers.llm_execution_agent.LLMExecutionAgent._invoke_llm",
            side_effect=model_response,
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.TokenBudgetTracker.tools_round_budget",
            new_callable=PropertyMock,
            return_value=1000,
        ),
    ):
        asyncio.run(initialize_rhokp_mcp(settings))
        payload = {
            "conversation_id": suid.get_suid(),
            "query": "openshift virtualization",
        }
        if endpoint == "/v1/streaming_query":
            payload["media_type"] = constants.MEDIA_TYPE_JSON
        response = client.post(endpoint, json=payload)

    assert response.status_code == 200
    if endpoint == "/v1/streaming_query":
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert events[-1]["event"] == "end"
        references = events[-1]["data"]["referenced_documents"]
    else:
        references = response.json()["referenced_documents"]
    assert [(doc["doc_url"], doc["doc_title"]) for doc in references] == [
        (url, "OpenShift Virtualization")
    ]
    assert rounds == 2
    remote_tool.ainvoke.assert_awaited_once_with(
        {
            "query": "openshift virtualization",
            "products": [
                {"product": "openshift_container_platform", "versions": ["4.19"]}
            ],
            "rows": config.ols_config.openshift_docs_mcp.max_results,
        }
    )


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_okp_tool_retrieval_without_tool_call_has_no_references(
    client: TestClient, endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return a successful answer without citations when the model skips the OKP tool."""
    config.ols_config.openshift_docs_mcp = OpenShiftDocsMCPSettings(
        url="https://lightspeed-rhokp.test.svc:8443/mcp"
    )
    monkeypatch.setattr(rhokp_mcp, "_RHOKP_SEARCH_TOOLS", {})
    config.ols_config.guardrails.tool_result_inspection.enabled = False
    remote_tool = MagicMock()
    remote_tool.name = "search"
    remote_tool.ainvoke = AsyncMock()
    mcp_client = MagicMock()
    mcp_client.get_tools = AsyncMock(return_value=[remote_tool])

    with (
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "ols.src.rag_index.rhokp_mcp.MultiServerMCPClient",
            return_value=mcp_client,
        ),
        patch(
            "ols.src.query_helpers.query_helper.load_llm",
            new=mock_llm_loader(mock_langchain_interface("answer")()),
        ),
    ):
        payload = {
            "conversation_id": suid.get_suid(),
            "query": "openshift virtualization",
        }
        if endpoint == "/v1/streaming_query":
            payload["media_type"] = constants.MEDIA_TYPE_JSON
        response = client.post(endpoint, json=payload)

    assert response.status_code == 200
    if endpoint == "/v1/streaming_query":
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert events[-1]["event"] == "end"
        references = events[-1]["data"]["referenced_documents"]
    else:
        references = response.json()["referenced_documents"]
    assert references == []
    mcp_client.get_tools.assert_awaited_once_with(server_name="rhokp")
    remote_tool.ainvoke.assert_not_awaited()
