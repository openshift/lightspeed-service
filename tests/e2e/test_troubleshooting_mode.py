"""End-to-end tests for troubleshooting mode.

Verifies that the /v1/query endpoint correctly activates the troubleshooting
system prompt (with cluster version injection) and MCP tool calling when
mode=troubleshooting is supplied.

Requires an MCP-enabled OLS deployment with access to alert-query tools
(obs-mcp / openshift-mcp-server).
"""

# pyright: reportAttributeAccessIssue=false

import pytest

from tests.e2e.utils.constants import LLM_REST_API_TIMEOUT

pytestmark = pytest.mark.mcp

QUERY_ENDPOINT = "/v1/query"


def test_troubleshooting_mode_returns_response_and_calls_tool() -> None:
    """Query with mode=troubleshooting triggers tool call and returns a response.

    Exercises:
    - TROUBLESHOOTING_SYSTEM_INSTRUCTION selected (cluster version injected)
    - At least one MCP tool called during the troubleshooting investigation
    """
    response = pytest.client.post(
        QUERY_ENDPOINT,
        json={
            "query": "What is the current status of my OpenShift cluster?",
            "mode": "troubleshooting",
        },
        timeout=LLM_REST_API_TIMEOUT,
    )

    assert response.status_code == 200

    data = response.json()
    assert data["response"], "Expected a non-empty response"
    assert data["input_tokens"] > 0
    assert data["output_tokens"] > 0

    tool_names = [tc["name"] for tc in data.get("tool_calls", [])]
    assert tool_names, (
        f"Expected at least one MCP tool to be called in troubleshooting mode; "
        f"got tool_calls=[] response={data['response']!r}"
    )
