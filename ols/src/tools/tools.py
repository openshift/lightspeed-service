"""Functions/Tools definition."""

from subprocess import run
from langchain.tools import tool


TIMEOUT = 20


@tool
def get_namespaces() -> str:
    """Fetch the list of all namespaces in the cluster"""
    output = run(["oc", "get", "namespaces"], capture_output=True, timeout=TIMEOUT)
    return output.stdout.decode("utf-8")


tools_map = {
    "get_namespaces": get_namespaces,
}
