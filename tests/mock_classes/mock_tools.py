"""Mocked tools for tool calling."""

from langchain_core.tools import StructuredTool


def get_namespaces_mock() -> str:
    """Mock function to return a list of namespaces in the cluster."""
    return """
NAME                                               STATUS   AGE
default                                            Active   25m
"""


tool = StructuredTool(
    name="get_namespaces_mock",
    description="Fetch the list of all namespaces in the cluster.",
    func=get_namespaces_mock,
    args_schema={},
    metadata={"readOnlyHint": True},
)

mock_tools_map = [tool]
