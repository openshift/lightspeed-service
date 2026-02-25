"""Mocked tools for tool calling."""

from langchain.tools import tool

NAMESPACES_OUTPUT = """
NAME                                               STATUS   AGE
default                                            Active   25m
"""


@tool
async def get_namespaces_mock() -> tuple[str, dict]:
    """Fetch the list of all namespaces in the cluster."""
    return NAMESPACES_OUTPUT, {}


mock_tools_map = [get_namespaces_mock]
