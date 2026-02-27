"""Mocked tools for tool calling."""

from typing import Any

from langchain.tools import tool
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel

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

MOCK_TOOL_META = {
    "ui": {
        "resourceUri": "ui://test-server/dashboard.html",
        "visibility": ["model", "app"],
    }
}


@tool
async def get_namespaces_mock() -> tuple[str, dict]:
    """Fetch the list of all namespaces in the cluster."""
    return NAMESPACES_OUTPUT, {}


@tool
async def get_pod_metrics_mock() -> tuple[str, dict]:
    """Fetch pod resource metrics for the cluster."""
    return "Pod utilization summary", {"structured_content": POD_STRUCTURED_CONTENT}


class _EmptySchema(BaseModel):
    pass


async def _namespaces_with_meta_coro(**kwargs: Any) -> tuple[str, dict]:
    return NAMESPACES_OUTPUT, {}


get_namespaces_with_meta_mock = StructuredTool(
    name="get_namespaces_with_meta_mock",
    description="Fetch the list of all namespaces in the cluster.",
    func=lambda **kwargs: (NAMESPACES_OUTPUT, {}),
    coroutine=_namespaces_with_meta_coro,
    response_format="content_and_artifact",
    args_schema=_EmptySchema,
    metadata={
        "_meta": MOCK_TOOL_META,
        "mcp_server": "test-server",
    },
)


mock_tools_map = [get_namespaces_mock]
mock_tools_with_structured_content = [get_pod_metrics_mock]
mock_tools_with_meta = [get_namespaces_with_meta_mock]
