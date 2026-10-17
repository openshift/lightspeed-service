"""Cluster tests for Classic tool-result inspection."""

# pyright: reportAttributeAccessIssue=false

import json
import re

import httpx
import pytest

from ols.src.tools.tool_result_inspection import TOOL_RESULT_SAFETY_FAILURE_MESSAGE
from ols.utils.suid import get_suid
from tests.e2e.mcp.server.server import (
    BENIGN_INSPECTION_RESULT,
    MALICIOUS_INSPECTION_RESULT,
)
from tests.e2e.utils import cluster
from tests.e2e.utils.constants import LLM_REST_API_TIMEOUT
from tests.e2e.utils.inspection_assertions import (
    assert_inspection_outcome,
    assert_rejected_result_absent,
    classifier_failures_since,
    format_inspection_failure_diagnostics,
)
from tests.e2e.utils.mcp_setup import NAMESPACE

pytestmark = pytest.mark.inspection

INSPECTION_HEADERS = {
    "mock-inspection": {"Authorization": "Bearer inspection-test-token"}
}
BENIGN_TOOL_NAME = "ols_inspection_benign"
DIAGNOSTIC_TOOL_NAME = "ols_inspection_diagnostic"
EVENTS_TOOL_NAME = "ols_inspection_events"


def _read_sse_events(response_text: str) -> list[dict]:
    """Decode JSON data events from an SSE response."""
    return [
        json.loads(line.removeprefix("data: "))
        for line in response_text.splitlines()
        if line.startswith("data: ")
    ]


def _assert_mock_tool_called(tool_name: str) -> None:
    """Make sure that the mock MCP server received the named tool call."""
    logs = cluster.run_oc(
        ["logs", "deployment/mcp-mock-server", "-n", NAMESPACE]
    ).stdout
    if f"MCP tool called: {tool_name}" not in logs:
        pytest.fail("The mock MCP server did not record the expected tool call")


def _service_logs() -> str:
    """Read app-server logs without exposing them in assertion failures."""
    return cluster.run_oc(
        ["logs", "deployment/lightspeed-app-server", "-n", NAMESPACE]
    ).stdout


def _inspection_outcomes_since(previous: str, current: str) -> list[tuple[str, str]]:
    """Find content-free classifier outcomes emitted during this query."""
    recent = current[len(previous) :] if current.startswith(previous) else ""
    return re.findall(
        r"Tool result inspection outcome=(benign|malicious|classifier_error) "
        r"result_type=(result|error)",
        recent,
    )


def _failure_diagnostics_since(
    previous_logs: str, status_code: int | None, tool_name: str
) -> str:
    """Collect only allowlisted service and mock-server failure metadata."""
    try:
        current_logs = _service_logs()
    except Exception:
        outcomes = None
        classifier_failures = None
    else:
        outcomes = _inspection_outcomes_since(previous_logs, current_logs)
        classifier_failures = classifier_failures_since(previous_logs, current_logs)
    try:
        mock_logs = cluster.run_oc(
            ["logs", "deployment/mcp-mock-server", "-n", NAMESPACE]
        ).stdout
        mock_tool_called = f"MCP tool called: {tool_name}" in mock_logs
    except Exception:
        mock_tool_called = None
    return format_inspection_failure_diagnostics(
        status_code, mock_tool_called, outcomes, classifier_failures
    )


def _post_with_inspection_diagnostics(
    endpoint: str,
    payload: dict[str, object],
    previous_logs: str,
    tool_name: str,
) -> httpx.Response:
    """Post an inspection query and report safe details on transport failure."""
    response: httpx.Response | None = None
    transport_failure: str | None = None
    try:
        response = pytest.client.post(
            endpoint, json=payload, timeout=LLM_REST_API_TIMEOUT
        )
    except httpx.TransportError as error:
        diagnostics = _failure_diagnostics_since(previous_logs, None, tool_name)
        transport_failure = (
            "The inspection query failed at the HTTP transport: "
            f"exception_type={type(error).__name__}; {diagnostics}"
        )
    if transport_failure is not None:
        pytest.fail(transport_failure)
    assert response is not None
    return response


def test_benign_tool_result_passes_inspection() -> None:
    """Accept benign mock tool output and return a successful query response."""
    previous_logs = _service_logs()
    response = _post_with_inspection_diagnostics(
        "/v1/query",
        {
            "query": (
                "Call the ols_inspection_benign tool and report its OpenShift status."
            ),
            "mcp_headers": INSPECTION_HEADERS,
        },
        previous_logs,
        BENIGN_TOOL_NAME,
    )
    if response.status_code != 200:
        diagnostics = _failure_diagnostics_since(
            previous_logs, response.status_code, BENIGN_TOOL_NAME
        )
        pytest.fail(f"The benign inspection query did not succeed: {diagnostics}")

    data = response.json()
    tool_names = [tool.get("name") for tool in data.get("tool_calls", [])]
    if BENIGN_TOOL_NAME not in tool_names:
        pytest.fail("The query did not call the benign inspection tool")
    if BENIGN_INSPECTION_RESULT not in json.dumps(data):
        pytest.fail("The successful response did not contain the benign tool result")

    _assert_mock_tool_called(BENIGN_TOOL_NAME)
    outcomes = _inspection_outcomes_since(previous_logs, _service_logs())
    assert_inspection_outcome([outcome for outcome, _ in outcomes], "benign")


@pytest.mark.parametrize(
    ("endpoint", "tool_name"),
    [
        ("/v1/query", DIAGNOSTIC_TOOL_NAME),
        ("/v1/streaming_query", DIAGNOSTIC_TOOL_NAME),
        ("/v1/query", EVENTS_TOOL_NAME),
    ],
    ids=["json", "sse", "tool-error"],
)
def test_malicious_tool_result_fails_closed_without_leaking_content(
    endpoint: str,
    tool_name: str,
) -> None:
    """Reject malicious output before the API, history, or logs expose it."""
    conversation_id = get_suid()
    query = (
        "Call the ols_inspection_events tool to get recent OpenShift cluster "
        "events, and report what it returns."
        if tool_name == EVENTS_TOOL_NAME
        else "Call the ols_inspection_diagnostic tool to get the current OpenShift "
        "cluster diagnostic status, and report what it returns."
    )
    payload = {
        "conversation_id": conversation_id,
        "query": query,
        "mcp_headers": INSPECTION_HEADERS,
    }
    if endpoint == "/v1/streaming_query":
        payload["media_type"] = "application/json"

    previous_logs = _service_logs()
    response = _post_with_inspection_diagnostics(
        endpoint, payload, previous_logs, tool_name
    )
    if endpoint == "/v1/query":
        if response.status_code != 500:
            pytest.fail("The non-streaming inspection query did not fail closed")
        if response.json() != {
            "detail": {
                "response": TOOL_RESULT_SAFETY_FAILURE_MESSAGE,
                "cause": "",
            }
        }:
            pytest.fail("The non-streaming query returned an unexpected failure")
        response_text = response.text
    else:
        if response.status_code != 200:
            pytest.fail(
                "The streaming inspection request did not return an SSE response"
            )
        events = _read_sse_events(response.text)
        tool_calls = [event for event in events if event.get("event") == "tool_call"]
        if not any(
            call.get("data", {}).get("name") == tool_name for call in tool_calls
        ):
            pytest.fail("The stream did not emit the malicious tool call")
        errors = [event for event in events if event.get("event") == "error"]
        if not any(
            event.get("data", {}).get("response") == TOOL_RESULT_SAFETY_FAILURE_MESSAGE
            for event in errors
        ):
            pytest.fail("The stream did not emit the fixed inspection failure")
        if any(event.get("event") == "tool_result" for event in events):
            pytest.fail("The stream emitted a rejected tool result")
        response_text = response.text

    history_response = pytest.client.get(
        f"/v1/conversations/{conversation_id}", timeout=LLM_REST_API_TIMEOUT
    )
    if history_response.status_code != 404:
        pytest.fail("The failed turn appeared in conversation history")

    _assert_mock_tool_called(tool_name)
    service_logs = _service_logs()
    mock_logs = cluster.run_oc(
        ["logs", "deployment/mcp-mock-server", "-n", NAMESPACE]
    ).stdout
    assert_rejected_result_absent(
        {
            "response_text": MALICIOUS_INSPECTION_RESULT in response_text,
            "history_response": MALICIOUS_INSPECTION_RESULT in history_response.text,
            "service_logs": MALICIOUS_INSPECTION_RESULT in service_logs,
            "mock_logs": MALICIOUS_INSPECTION_RESULT in mock_logs,
        }
    )
    outcomes = _inspection_outcomes_since(previous_logs, service_logs)
    assert_inspection_outcome([outcome for outcome, _ in outcomes], "malicious")
    assert (
        tool_name != EVENTS_TOOL_NAME or ("malicious", "error") in outcomes
    ), "The classifier did not reject the tool error result"
