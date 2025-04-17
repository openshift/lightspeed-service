"""test tool."""

import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("openshift")


@mcp.tool()
async def oc_get(oc_get_args: list[str]) -> str:
    """Display one or many resources from OpenShift cluster.

    Standard `oc` flags and options are valid.

    Args:
        oc_get_args: Arguments for oc get

    Examples:
        # List all pods in ps output format.
        oc get pods

        # List all pods in ps output format with more information (such as node name).
        oc get pods -o wide

        # List events for cluster
        oc get events

        # List events for namespace
        oc get events -n namespace

        # List a single replication controller with specified NAME in ps output format.
        oc get replicationcontroller web

        # List deployments in JSON output format, in the "v1" version of the "apps" API group:
        oc get deployments.v1.apps -o json

        # List a pod identified by type and name specified in "pod.yaml" in JSON output format.
        oc get -f pod.yaml -o json

        # List all replication controllers and services together in ps output format.
        oc get rc,services
    """
    # token acquiring happens in central run_oc function in real code
    return f"pod1, pod2 with token {os.environ['OC_USER_TOKEN']}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
