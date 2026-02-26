"""Mocked tools for tool calling."""

from langchain.tools import tool

NAMESPACES_OUTPUT = """
NAME                                               STATUS   AGE
default                                            Active   25m
"""

POD_STRUCTURED_CONTENT = {
    "pods": [
        {"name": "pod1", "cpu": 45, "memory": 62},
        {"name": "pod2", "cpu": 12, "memory": 34},
    ],
    "summary": {"totalPods": 2, "avgCpu": 28, "avgMemory": 48},
}


@tool
async def get_namespaces_mock() -> tuple[str, dict]:
    """Fetch the list of all namespaces in the cluster."""
    return NAMESPACES_OUTPUT, {}


@tool
async def get_pod_metrics_mock() -> tuple[str, dict]:
    """Fetch pod resource metrics for the cluster."""
    return "Pod utilization summary", {"structured_content": POD_STRUCTURED_CONTENT}


mock_tools_map = [get_namespaces_mock]
mock_tools_with_structured_content = [get_pod_metrics_mock]
