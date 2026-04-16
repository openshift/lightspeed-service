r"""End-to-end tests for MCP integration.

These tests run against a live OLS instance and a mock MCP server.
The mock MCP server must be reachable from OLS (locally or on-cluster).

On-cluster: the mock server is deployed as a pod by conftest (OLS_CONFIG_SUFFIX=mcp).
Locally: start the mock server manually (python tests/e2e/mcp/server/server.py 3000).
"""

# pyright: reportAttributeAccessIssue=false

import json
import threading
from typing import Optional

import pytest

from tests.e2e.utils.constants import LLM_REST_API_TIMEOUT

pytestmark = pytest.mark.mcp

CLIENT_HEADERS = {"mock-client-auth": {"Authorization": "Bearer my-client-token-456"}}

_MCP_DETAIL_RESPONSE_CHARS = 4000
_MCP_DETAIL_STREAMING_CHARS = 12000


def _query_response_excerpt(data: dict) -> str:
    """Return log-safe excerpt of /v1/query JSON for assertion failures."""
    parts: list[str] = []
    response_text = data.get("response")
    if response_text is not None:
        text = str(response_text)
        if len(text) > _MCP_DETAIL_RESPONSE_CHARS:
            text = (
                f"{text[:_MCP_DETAIL_RESPONSE_CHARS]}..."
                f"({len(str(response_text))} chars total)"
            )
        parts.append(f"response={text!r}")
    parts.append(f"tool_calls={json.dumps(data.get('tool_calls'), default=str)}")
    if data.get("tool_results") is not None:
        parts.append(
            f"tool_results={json.dumps(data.get('tool_results'), default=str)}"
        )
    return " | ".join(parts)


def _streaming_events_excerpt(events: list[dict]) -> str:
    """Return log-safe excerpt of streaming SSE events for assertion failures."""
    blob = json.dumps(events, default=str)
    full_len = len(blob)
    if full_len > _MCP_DETAIL_STREAMING_CHARS:
        return f"{blob[:_MCP_DETAIL_STREAMING_CHARS]}...({full_len} chars total)"
    return blob


def _query(
    query: str,
    mcp_headers: Optional[dict[str, dict[str, str]]] = None,
) -> dict:
    """Send a query to OLS and return the parsed JSON response."""
    body: dict = {"query": query}
    if mcp_headers is not None:
        body["mcp_headers"] = mcp_headers
    response = pytest.client.post(
        "/v1/query",
        json=body,
        timeout=LLM_REST_API_TIMEOUT,
    )
    assert response.status_code == 200
    return response.json()


def _assert_tool_called(data: dict, tool_name: str) -> None:
    """Assert that the named tool appears in tool_calls."""
    tool_names = [tc["name"] for tc in data.get("tool_calls", [])]
    if tool_name not in tool_names:
        raise AssertionError(
            f"Expected tool {tool_name!r} in {tool_names}. {_query_response_excerpt(data)}"
        )


def _assert_no_tools(data: dict) -> None:
    """Assert that no tools were called."""
    tcalls = data.get("tool_calls", [])
    if tcalls != []:
        names = [tc.get("name") for tc in tcalls]
        raise AssertionError(
            f"Expected no tool_calls, got {names}. {_query_response_excerpt(data)}"
        )


# ---------------------------------------------------------------------------
# Test 1 - MCP configuration
# ---------------------------------------------------------------------------


def test_discovery_endpoint_lists_client_auth_servers() -> None:
    """GET /v1/mcp/client-auth-headers returns only servers needing client headers."""
    response = pytest.client.get(
        "/v1/mcp/client-auth-headers", timeout=LLM_REST_API_TIMEOUT
    )
    assert response.status_code == 200
    data = response.json()

    server_names = [s["server_name"] for s in data["servers"]]
    assert "mock-client-auth" in server_names

    for server in data["servers"]:
        if server["server_name"] == "mock-client-auth":
            assert "Authorization" in server["required_headers"]


def test_wrong_url_graceful_degradation() -> None:
    """OLS handles an unreachable MCP server (mock-bad-url) without crashing."""
    _query("list available tools")


# ---------------------------------------------------------------------------
# Test 2 - Query without client headers
# ---------------------------------------------------------------------------


def test_query_without_client_headers() -> None:
    """Without mcp_headers file-auth and k8s-auth tools work, client-auth doesn't."""
    _assert_tool_called(
        _query("check my openshift cluster status"),
        "openshift_cluster_status",
    )
    _assert_tool_called(
        _query("get openshift pod logs for my-app in namespace default"),
        "openshift_pod_logs",
    )
    _assert_no_tools(
        _query("use openshift_route_info to get route details for my-app"),
    )


# ---------------------------------------------------------------------------
# Test 3 - Query with client headers
# ---------------------------------------------------------------------------


def test_query_with_client_headers() -> None:
    """With mcp_headers the client-auth server tool is invocable."""
    _assert_tool_called(
        _query(
            "use openshift_route_info to get route details for my-app",
            mcp_headers=CLIENT_HEADERS,
        ),
        "openshift_route_info",
    )
    _assert_tool_called(
        _query("check my openshift cluster status", mcp_headers=CLIENT_HEADERS),
        "openshift_cluster_status",
    )
    _assert_tool_called(
        _query(
            "get openshift pod logs for my-app in namespace default",
            mcp_headers=CLIENT_HEADERS,
        ),
        "openshift_pod_logs",
    )


# ---------------------------------------------------------------------------
# Streaming / approval helpers
# ---------------------------------------------------------------------------


def _stream_events(
    query: str,
    mcp_headers: Optional[dict[str, dict[str, str]]] = None,
) -> list[dict]:
    """Send a streaming query and collect all SSE events as dicts."""
    body: dict = {"query": query, "media_type": "application/json"}
    if mcp_headers is not None:
        body["mcp_headers"] = mcp_headers
    events: list[dict] = []
    with pytest.client.stream(
        "POST",
        "/v1/streaming_query",
        json=body,
        timeout=LLM_REST_API_TIMEOUT,
    ) as resp:
        assert resp.status_code == 200
        events.extend(
            json.loads(line[len("data: ") :])
            for line in resp.iter_lines()
            if line.startswith("data: ")
        )
    return events


def _approve_async(approval_id: str, approved: bool) -> None:
    """Submit an approval decision from a background thread."""
    resp = pytest.client.post(
        "/v1/tool-approvals/decision",
        json={"approval_id": approval_id, "approved": approved},
        timeout=LLM_REST_API_TIMEOUT,
    )
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Test 4 - Tool approval with tool_annotations strategy
# ---------------------------------------------------------------------------


def test_readonly_tool_skips_approval() -> None:
    """Tool with readOnlyHint=true executes without an approval_required event."""
    events = _stream_events("check my openshift cluster status")
    event_types = [e.get("event") for e in events]
    assert "approval_required" not in event_types
    tool_calls = [e for e in events if e.get("event") == "tool_call"]
    tool_names = [tc["data"]["name"] for tc in tool_calls]
    if "openshift_cluster_status" not in tool_names:
        raise AssertionError(
            "Expected openshift_cluster_status in streaming tool_call events; "
            f"tool_names={tool_names}, event_types={event_types}, "
            f"events={_streaming_events_excerpt(events)}"
        )


def _stream_with_approval(
    query: str,
    approved: bool,
    mcp_headers: Optional[dict[str, dict[str, str]]] = None,
) -> list[dict]:
    """Stream a query, respond to the first approval_required event, return all events."""
    body: dict = {"query": query, "media_type": "application/json"}
    if mcp_headers is not None:
        body["mcp_headers"] = mcp_headers
    events: list[dict] = []
    decision_sent = threading.Event()

    def run() -> None:
        with pytest.client.stream(
            "POST",
            "/v1/streaming_query",
            json=body,
            timeout=LLM_REST_API_TIMEOUT,
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[len("data: ") :])
                events.append(event)
                if (
                    event.get("event") == "approval_required"
                    and not decision_sent.is_set()
                ):
                    decision_sent.set()
                    _approve_async(event["data"]["approval_id"], approved=approved)

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=LLM_REST_API_TIMEOUT)
    assert not t.is_alive(), "Streaming request did not complete in time"
    return events


def test_tool_no_annotation_approved() -> None:
    """Tool with no annotations requires approval; approving lets it execute."""
    events = _stream_with_approval(
        "get openshift pod logs for my-app in namespace default",
        approved=True,
    )
    event_types = [e.get("event") for e in events]
    assert "approval_required" in event_types
    assert "tool_result" in event_types


def test_tool_readonly_false_approved() -> None:
    """Tool with readOnlyHint=false requires approval; approving lets it execute."""
    events = _stream_with_approval(
        "use openshift_route_info to get route details for my-app",
        approved=True,
        mcp_headers=CLIENT_HEADERS,
    )
    event_types = [e.get("event") for e in events]
    assert "approval_required" in event_types
    assert "tool_result" in event_types


def test_tool_approval_denied() -> None:
    """Denying approval emits a rejection tool_result with error status."""
    events = _stream_with_approval(
        "get openshift pod logs for my-app in namespace default",
        approved=False,
    )
    event_types = [e.get("event") for e in events]
    assert "approval_required" in event_types
    tool_results = [e["data"] for e in events if e.get("event") == "tool_result"]
    rejected = [r for r in tool_results if r.get("status") == "error"]
    assert len(rejected) > 0, f"Expected a rejected tool_result, got: {tool_results}"


def test_tool_approval_timeout() -> None:
    """Not responding to approval_required within the timeout produces an error tool_result."""
    events = _stream_events(
        "get openshift pod logs for my-app in namespace default",
    )
    event_types = [e.get("event") for e in events]
    assert "approval_required" in event_types
    tool_results = [e["data"] for e in events if e.get("event") == "tool_result"]
    timed_out = [r for r in tool_results if r.get("status") == "error"]
    assert len(timed_out) > 0, f"Expected a timed-out tool_result, got: {tool_results}"
