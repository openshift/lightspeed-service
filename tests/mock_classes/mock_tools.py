"""Mocked tools for tool calling."""

from langchain.tools import tool


# Define Mock/Sample tools
@tool
def get_namespaces_mock() -> tuple[str, str]:
    """Fetch the list of all namespaces in the cluster."""
    return (
        "success",
        """
NAME                                               STATUS   AGE
default                                            Active   25m
""",
    )


mock_tools_map = {"get_namespaces_mock": get_namespaces_mock}
