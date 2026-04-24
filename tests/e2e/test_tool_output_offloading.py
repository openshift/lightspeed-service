"""End-to-end tests for tool output offloading.

These tests verify that large MCP tool outputs are offloaded to disk
and that the LLM uses the search/read retrieval tools to access them.

Requirements:
  - MCP server with tools that produce large outputs (e.g., events_list,
    pods_list, alertmanager_alerts from openshift-mcp-server)
  - max_tokens_per_tool_output set low enough to trigger offloading
    (e.g., 500 in model parameters)
"""

# pyright: reportAttributeAccessIssue=false

import json

import pytest
import requests

from ols import constants
from ols.utils import suid
from tests.e2e.utils import response as response_utils
from tests.e2e.utils.constants import LLM_REST_API_TIMEOUT
from tests.e2e.utils.decorators import retry

STREAMING_QUERY_ENDPOINT = "/v1/streaming_query"


def post_with_defaults(endpoint, **kwargs):
    """Send POST request with HTTP/1.0 header and timeout."""
    return pytest.client.post(
        endpoint,
        headers={"Connection": "close"},
        timeout=kwargs.pop("timeout", LLM_REST_API_TIMEOUT),
        **kwargs,
    )


def parse_streaming_response_to_events(response_text: str) -> list[dict]:
    """Parse SSE streaming response into a list of event dicts."""
    json_objects = [
        line.replace("data: ", "") for line in response_text.split("\n") if line.strip()
    ]
    json_array = "[" + ",".join(json_objects) + "]"
    return json.loads(json_array)


def extract_tool_events(
    events: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Extract tool_call and tool_result events from the event stream."""
    tool_calls = [e for e in events if e.get("event") == "tool_call"]
    tool_results = [e for e in events if e.get("event") == "tool_result"]
    return tool_calls, tool_results


@pytest.mark.tool_calling
@retry(max_attempts=3, wait_between_runs=10)
def test_offloaded_tool_output_triggers_retrieval() -> None:
    """Verify large tool outputs are offloaded and retrieval tools are invoked.

    Sends a broad query that should trigger MCP tools returning output
    exceeding max_tokens_per_tool_output. Asserts that:
      1. At least one tool_result contains the offloading placeholder
      2. The model calls search_offloaded_content or read_offloaded_content
    """
    cid = suid.get_suid()
    response = post_with_defaults(
        STREAMING_QUERY_ENDPOINT,
        json={
            "conversation_id": cid,
            "query": (
                "List all events across all namespaces and find any "
                "warnings or errors that need attention"
            ),
            "media_type": constants.MEDIA_TYPE_JSON,
        },
        timeout=180,
    )
    assert response.status_code == requests.codes.ok
    response_utils.check_content_type(response, constants.MEDIA_TYPE_JSON)

    events = parse_streaming_response_to_events(response.text)
    tool_calls, tool_results = extract_tool_events(events)

    assert tool_calls, "Expected at least one tool_call event"
    assert tool_results, "Expected at least one tool_result event"

    offloaded_results = [
        tr
        for tr in tool_results
        if "[Offloaded:" in str(tr.get("data", {}).get("content", ""))
    ]
    if not offloaded_results:
        pytest.skip(
            "No tool output was offloaded — tool outputs fit within the "
            f"per-tool token budget on this cluster ({len(tool_results)} "
            "tool_result(s), none exceeded budget). Lower tool_budget_ratio "
            "or increase cluster activity to trigger offloading."
        )

    placeholder = str(offloaded_results[0]["data"]["content"])
    assert "ref_id:" in placeholder
    assert "search_offloaded_content" in placeholder

    retrieval_tool_names = {"search_offloaded_content", "read_offloaded_content"}
    tool_call_names = {tc.get("data", {}).get("name", "") for tc in tool_calls}
    retrieval_calls = tool_call_names & retrieval_tool_names
    assert retrieval_calls, (
        f"Expected retrieval tool calls (search/read_offloaded_content), "
        f"but model only called: {sorted(tool_call_names)}"
    )


@pytest.mark.tool_calling
@retry(max_attempts=3, wait_between_runs=10)
def test_offloaded_content_placeholder_structure() -> None:
    """Verify the offloading placeholder contains expected metadata.

    Checks that the placeholder includes ref_id, line count, byte size,
    and instructions for both retrieval tools.
    """
    cid = suid.get_suid()
    response = post_with_defaults(
        STREAMING_QUERY_ENDPOINT,
        json={
            "conversation_id": cid,
            "query": "Show me all pods running across the entire cluster",
            "media_type": constants.MEDIA_TYPE_JSON,
        },
        timeout=180,
    )
    assert response.status_code == requests.codes.ok

    events = parse_streaming_response_to_events(response.text)
    _, tool_results = extract_tool_events(events)

    offloaded_results = [
        tr
        for tr in tool_results
        if "[Offloaded:" in str(tr.get("data", {}).get("content", ""))
    ]
    if not offloaded_results:
        pytest.skip(
            "No tool output was offloaded — tool outputs may be below "
            "max_tokens_per_tool_output threshold for this cluster"
        )

    placeholder = str(offloaded_results[0]["data"]["content"])
    assert "ref_id:" in placeholder, "Placeholder missing ref_id"
    assert "lines:" in placeholder, "Placeholder missing line count"
    assert "bytes" in placeholder, "Placeholder missing byte size"
    assert "search_offloaded_content" in placeholder
    assert "read_offloaded_content" in placeholder
