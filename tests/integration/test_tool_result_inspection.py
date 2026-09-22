"""Integration tests for Classic tool-result inspection."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
import requests
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from ols import config, constants
from ols.src.tools.tool_result_inspection import (
    TOOL_RESULT_SAFETY_FAILURE_MESSAGE,
    ToolResultInspectionError,
    ToolResultRejectedError,
)
from ols.utils import suid
from tests.mock_classes.mock_llm_loader import mock_llm_loader
from tests.mock_classes.mock_tools import mock_tools_map


@pytest.fixture(scope="function")
def client() -> TestClient:
    """Create a client with the integration configuration."""
    config.reload_from_yaml_file("tests/config/config_for_integration_tests.yaml")
    from ols.app.main import app  # pylint: disable=import-outside-toplevel

    return TestClient(app)


async def _chunks(*chunks: AIMessageChunk):
    """Yield model stream chunks in order."""
    for chunk in chunks:
        yield chunk


def _tool_calling_side_effect() -> object:
    """Return a two-round model stream that invokes the mock tool."""
    calls = 0

    def side_effect(*args, **kwargs):
        """Return a tool-call round followed by a final-answer round."""
        nonlocal calls
        calls += 1
        if calls == 1:
            return _chunks(
                AIMessageChunk(
                    content="",
                    response_metadata={"finish_reason": "tool_calls"},
                    tool_call_chunks=[
                        {
                            "name": "get_namespaces_mock",
                            "args": "{}",
                            "id": "call_id1",
                            "index": 0,
                        }
                    ],
                )
            )
        return _chunks(
            AIMessageChunk(content="final answer", response_metadata={}),
            AIMessageChunk(content="", response_metadata={"finish_reason": "stop"}),
        )

    return side_effect


def _tool_loop_patches(classifier: MagicMock, invoke: MagicMock):
    """Patch the MCP and model seams while preserving the real endpoint flow."""
    mcp_servers = {"fake-server": {"transport": "http", "url": "http://fake-server"}}
    mock_config = MagicMock()
    mock_config.tools_rag = None
    mock_config.mcp_servers.servers = [MagicMock()]
    return (
        patch.multiple(
            "ols.utils.mcp_utils",
            config=mock_config,
            _gather_and_populate_tools=AsyncMock(
                return_value=(mcp_servers, mock_tools_map)
            ),
            MultiServerMCPClient=MagicMock(),
        ),
        patch(
            "ols.src.query_helpers.llm_execution_agent.LLMExecutionAgent._invoke_llm",
            new=invoke,
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.TokenBudgetTracker.tools_round_budget",
            new_callable=PropertyMock,
            return_value=1000,
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.ToolResultClassifier",
            return_value=classifier,
        ),
        patch("ols.src.query_helpers.query_helper.load_llm", new=mock_llm_loader(None)),
    )


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_rejected_tool_result_uses_fixed_safety_response(
    client: TestClient, endpoint: str
) -> None:
    """Reject tool output before it reaches the response path."""
    classifier = MagicMock()
    classifier.inspect = AsyncMock(
        side_effect=ToolResultRejectedError("classifier details must stay private")
    )
    invoke = MagicMock(side_effect=_tool_calling_side_effect())

    patches = _tool_loop_patches(classifier, invoke)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        payload = {"conversation_id": suid.get_suid(), "query": "list namespaces"}
        if endpoint == "/v1/streaming_query":
            payload["media_type"] = constants.MEDIA_TYPE_TEXT
        response = client.post(endpoint, json=payload)

    if endpoint == "/v1/streaming_query":
        assert response.status_code == requests.codes.ok
        assert TOOL_RESULT_SAFETY_FAILURE_MESSAGE in response.text
        assert '"event": "tool_result"' not in response.text
    else:
        assert response.status_code == requests.codes.internal_server_error
        assert response.json() == {
            "detail": {
                "response": TOOL_RESULT_SAFETY_FAILURE_MESSAGE,
                "cause": "",
            }
        }
    assert "classifier details" not in response.text
    assert "NAME" not in response.text


def test_classifier_failure_uses_fixed_safety_response(client: TestClient) -> None:
    """Fail closed when the classifier cannot complete inspection."""
    classifier = MagicMock()
    classifier.inspect = AsyncMock(
        side_effect=ToolResultInspectionError("provider failure details")
    )
    invoke = MagicMock(side_effect=_tool_calling_side_effect())

    patches = _tool_loop_patches(classifier, invoke)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        response = client.post(
            "/v1/query",
            json={"conversation_id": suid.get_suid(), "query": "list namespaces"},
        )

    assert response.status_code == requests.codes.internal_server_error
    assert response.json() == {
        "detail": {
            "response": TOOL_RESULT_SAFETY_FAILURE_MESSAGE,
            "cause": "",
        }
    }
    assert "provider failure details" not in response.text


def test_disabled_inspection_preserves_tool_flow(client: TestClient) -> None:
    """Skip classifier calls when inspection is disabled."""
    config.ols_config.guardrails.tool_result_inspection.enabled = False
    classifier = MagicMock()
    classifier.inspect = AsyncMock()
    invoke = MagicMock(side_effect=_tool_calling_side_effect())

    patches = _tool_loop_patches(classifier, invoke)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        response = client.post(
            "/v1/query",
            json={"conversation_id": suid.get_suid(), "query": "list namespaces"},
        )

    assert response.status_code == requests.codes.ok
    assert response.json()["response"] == "final answer"
    classifier.inspect.assert_not_called()
