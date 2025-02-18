"""Functions/Tools definition."""

from langchain.tools import tool


# TODO: Sample function to test the flow. Add actual tools/functions here
@tool
def get_namespaces() -> str:
    """Fetch the list of all namespaces in the cluster."""
    return """
NAME                                               STATUS   AGE
default                                            Active   25m
namespace1                                         Active   25m
"""


tools_map = {
    "get_namespaces": get_namespaces,
}
