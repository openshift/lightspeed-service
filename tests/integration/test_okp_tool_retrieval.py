"""Integration tests for OKP documentation tool retrieval through query endpoints."""

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from ols import config, constants
from ols.app.models.config import SolrHybridSettings
from ols.src.rag_index.solr_support import RetrievedChunk, SolrHybridSearch
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
    solr = MagicMock(spec=SolrHybridSearch)
    solr.search = AsyncMock(
        return_value=[
            RetrievedChunk(
                text="OpenShift Virtualization runs virtual machines.",
                score=3.5,
                metadata={"title": "OpenShift Virtualization", "docs_url": url},
            )
        ]
    )
    config.ols_config.solr_hybrid = SolrHybridSettings()
    monkeypatch.setattr(config, "_cached_solr_hybrid_search", solr)
    monkeypatch.setattr(config, "_solr_hybrid_initialized", True)
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
                        "name": "search_openshift_documentation",
                        "args": '{"search_query": "openshift virtualization"}',
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
    solr.search.assert_awaited_once()
    assert solr.search.await_args.args == ("openshift virtualization",)


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_okp_tool_retrieval_without_tool_call_has_no_references(
    client: TestClient, endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return a successful answer without citations when the model skips the OKP tool."""
    solr = MagicMock(spec=SolrHybridSearch)
    solr.search = AsyncMock()
    config.ols_config.solr_hybrid = SolrHybridSettings()
    monkeypatch.setattr(config, "_cached_solr_hybrid_search", solr)
    monkeypatch.setattr(config, "_solr_hybrid_initialized", True)
    config.ols_config.guardrails.tool_result_inspection.enabled = False

    with (
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new_callable=AsyncMock,
            return_value=[],
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
    solr.search.assert_not_awaited()


def test_okp_tool_retrieval_does_not_leak_solr_client() -> None:
    """Leave shared Solr cache fields untouched after endpoint tests."""
    assert config._cached_solr_hybrid_search is None
    assert config._solr_hybrid_initialized is False
