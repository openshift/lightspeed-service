"""End-to-end tests for obs-mcp metrics tools in troubleshooting mode.

Verifies that the real openshift-mcp-server (with the observability/metrics
toolset enabled) calls get_alerts and list_metrics when mode=troubleshooting
is used.

Requirements:
  - OLS deployed with introspectionEnabled: true (tool_calling suite)
  - openshift-mcp-server running with observability/metrics toolset
  - Alertmanager and Thanos Querier reachable inside the cluster
"""

# pyright: reportAttributeAccessIssue=false

import os

import pytest

from tests.e2e.utils.constants import LLM_REST_API_TIMEOUT
from tests.e2e.utils.decorators import retry

pytestmark = pytest.mark.tool_calling

QUERY_ENDPOINT = "/v1/query"

_METRICS_TIMEOUT = max(LLM_REST_API_TIMEOUT, 180)


@pytest.mark.xfail(
    condition=os.getenv("PROVIDER", "").startswith("bedrock"),
    strict=False,
    raises=Exception,
    reason=(
        "DeepSeek (Bedrock) calls get_alerts without the Watchdog filter, "
        "returning all Alertmanager alerts and crashing the OLS worker "
        "(RemoteProtocolError). OLS-4301 partially fixed this; full fix pending."
    ),
)
def test_troubleshooting_mode_calls_get_alerts() -> None:
    """Querying the Watchdog alert status triggers the get_alerts tool.

    Watchdog is a standard heartbeat alert always firing on OpenShift clusters.
    Asking specifically about it causes the LLM to call get_alerts with a tight
    filter (alertname=Watchdog), returning exactly one small alert record.
    """
    response = pytest.client.post(
        QUERY_ENDPOINT,
        json={
            "query": "Is the Watchdog alert currently firing in the cluster?",
            "mode": "troubleshooting",
        },
        timeout=_METRICS_TIMEOUT,
    )

    assert response.status_code == 200

    data = response.json()
    assert data["response"], "Expected a non-empty response"
    assert data["input_tokens"] > 0
    assert data["output_tokens"] > 0

    tool_names = [tc["name"] for tc in data.get("tool_calls", [])]
    assert "get_alerts" in tool_names, (
        f"Expected get_alerts to be called; got tool_calls={tool_names!r}, "
        f"response={data['response']!r}"
    )


@retry(max_attempts=3, wait_between_runs=10)
def test_troubleshooting_mode_calls_list_metrics() -> None:
    """Querying a cluster metric triggers the obs-mcp metrics investigation workflow.

    list_metrics is the mandatory first step of any metric query per the obs-mcp
    server prompt. Asserting it was called proves the observability/metrics toolset
    is active and OLS correctly routes metric queries to openshift-mcp-server.

    Note: execute_instant_query requires cluster-monitoring-view RBAC on the test
    service account. Since that role is not granted in the tool_calling CI suite,
    Prometheus calls return 403 and the LLM stops after list_metrics. Asserting
    list_metrics is therefore the stable and correct signal here.
    """
    response = pytest.client.post(
        QUERY_ENDPOINT,
        json={
            "query": "What Prometheus metrics are available for monitoring CPU usage?",
            "mode": "troubleshooting",
        },
        timeout=_METRICS_TIMEOUT,
    )

    assert response.status_code == 200

    data = response.json()
    assert data["response"], "Expected a non-empty response"
    assert data["input_tokens"] > 0
    assert data["output_tokens"] > 0

    tool_names = [tc["name"] for tc in data.get("tool_calls", [])]
    assert "list_metrics" in tool_names, (
        f"Expected list_metrics to be called; got tool_calls={tool_names!r}, "
        f"response={data['response']!r}"
    )
