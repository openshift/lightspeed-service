"""OpenShift CLI tools."""

# Need this to avoid adding token arg description to tool doc string.
# inline ignore is getting removed.
# ruff: noqa: D417

# Debug note: to debug the tools in local service instance, eg. against
# cluster, use the KUBECONFIG env variable to point to the kubeconfig
# file with the cluster configuration, as the oc CLI checks this when no
# server is provided in the command.


import logging
import subprocess
from typing import Annotated

from langchain.tools import tool
from langchain_core.tools import InjectedToolArg
from langchain_core.tools.base import BaseTool

from ols.plugins.tools.tools import (
    ToolsContext,
    ToolSetProvider,
    register_tool_provider_as,
)

logger = logging.getLogger(__name__)


# TODO: OLS-1443
# TODO: might also sanitize the case when llm sends the full command instead
# of just args, eg. [oc get pods -n bla] -> [pods -n bla]
def sanitize_oc_args(args: list[str]) -> list[str]:
    """Sanitize `oc` CLI arguments."""
    blocked_chars = {";", "&", "|", "`", "$", "(", ")", "<", ">", "\\"}

    sanitized_args = []
    for arg in args:
        safe_part = []
        for char in arg:
            if char in blocked_chars:
                # stop processing further characters in this argument
                logger.warning(
                    "Problematic character(s) found in oc tool argument '%s'", arg
                )
                break
            safe_part.append(char)

        if safe_part:  # only add non-empty results
            sanitized_args.append("".join(safe_part).strip())

    # Temporary workaround for granite.
    # Sometimes within the list we may get two args combined; ex: [top pod]
    sanitized_args = " ".join(sanitized_args).split(" ")
    # Sometimes model gives args which are already added to the tool.
    remove_arg = ["oc", "get", "describe", "logs", "status", "adm", "top"]
    for arg in remove_arg:
        if arg in sanitized_args:
            sanitized_args.remove(arg)

    return sanitized_args


def run_oc(args: list[str]) -> subprocess.CompletedProcess:
    """Run `oc` CLI with provided arguments and command."""
    res = subprocess.run(  # noqa: S603
        ["oc", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        shell=False,
    )
    return res


@tool
def oc_get(oc_get_args: list[str], token: Annotated[str, InjectedToolArg]) -> str:
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
    result = run_oc(["get", *sanitize_oc_args(oc_get_args), "--token", token])
    return result.stdout


@tool
def oc_describe(
    oc_describe_args: list[str], token: Annotated[str, InjectedToolArg]
) -> str:
    """Show details of a specific resource or group of resources.

    Print a detailed description of the selected resources, including related resources such as events or controllers.
    You may select a single object by name, all objects of that type, provide a name prefix, or label selector.

    Args:
        oc_describe_args: Arguments for oc describe

    Examples:
        # Describe a node
        oc describe nodes kubernetes-node-emt8.c.myproject.internal

        # Describe a pod
        oc describe pods/nginx

        # Describe a pod identified by type and name in "pod.json"
        oc describe -f pod.json

        # Describe all pods
        oc describe pods

        # Describe pods by label name=myLabel
        oc describe po -l name=myLabel

        # Describe all pods managed by the 'frontend' replication controller
        oc describe pods frontend
    """  # noqa: E501
    result = run_oc(["describe", *sanitize_oc_args(oc_describe_args), "--token", token])
    return result.stdout


@tool
def oc_logs(oc_logs_args: list[str], token: Annotated[str, InjectedToolArg]) -> str:
    """Print the logs for a resource.

    Supported resources are builds, build configs (bc), deployment configs (dc), and pods.
    When a pod is specified and has more than one container, the container name should be specified via -c.
    When a build config or deployment config is specified, you can view the logs for a particular version of it via --version.

    Args:
        oc_logs_args: Arguments for oc logs

    Examples:
        # Start streaming the logs of the most recent build of the openldap build config.
        oc logs -f bc/openldap

        # Get the logs of the first deployment for the mysql deployment config.
        oc logs --version=1 dc/mysql

        # Return a snapshot of ruby-container logs from pod backend.
        oc logs backend -c ruby-container

        # Start streaming of ruby-container logs from pod backend.
        oc logs -f pod/backend -c ruby-container
    """  # noqa: E501
    result = run_oc(["logs", *sanitize_oc_args(oc_logs_args), "--token", token])
    return result.stdout


@tool
def oc_status(oc_status_args: list[str], token: Annotated[str, InjectedToolArg]) -> str:
    """Show a high level overview of the current project.

    This command will show services, deployment configs, build configurations, & active deployments.
    If you have any misconfigured components information about them will be shown.
    For more information about individual items, use the describe command \
    (e.g. oc describe buildconfig, oc describe deploymentconfig, oc describe service).

    Args:
        oc_status_args: Arguments for oc status

    Examples:
        # See an overview of the current project.
        oc status

        # Export the overview of the current project in an svg file.
        oc status -o dot | dot -T svg -o project.svg

        # See an overview of the current project including details for any identified issues.
        oc --suggest
    """
    result = run_oc(["status", *sanitize_oc_args(oc_status_args), "--token", token])
    return result.stdout


@tool
def show_pods(token: Annotated[str, InjectedToolArg]) -> str:
    """Show resource usage (CPU and memory) for all pods accross all namespaces.

    Usecases:
        - Pods resource usage monitoring.
        - Resource allocation monitoring.
        - Average resources consumption.

    The output format is:
        NAMESPACE    NAME                                              CPU(cores)  MEMORY(bytes)
        kube-system  konnectivity-agent-qwnsd                          1m          24Mi
        kube-system  kube-apiserver-proxy-ip-10-0-130-91.ec2.internal  2m          13Mi
    """
    result = run_oc([*["adm", "top", "pods", "-A"], "--token", token])
    return result.stdout


@tool
def oc_adm_top(
    oc_adm_top_args: list[str], token: Annotated[str, InjectedToolArg]
) -> str:
    """Show usage statistics of resources on the server.

    This command analyzes resources managed by the platform and presents current usage statistics.

    When no options are provided, the command will list given resource in default namespace.
    To get the resources across namespaces, use `-A` flag.

    Args:
        oc_adm_top_args: Arguments for oc adm top

    Usage:
        oc adm top [commands] [options]

    Available Commands:
        images       Show usage statistics for Images
        imagestreams Show usage statistics for ImageStreams
        node         Display Resource (CPU/Memory/Storage) usage of nodes
        pod          Display Resource (CPU/Memory/Storage) usage of pods

    Options:
        --namespace <namespace>
            Lists resources for specified namespace.
    """
    result = run_oc(
        ["adm", "top", *sanitize_oc_args(oc_adm_top_args), "--token", token]
    )
    return result.stdout


@register_tool_provider_as("openshift")
class OCToolProvider(ToolSetProvider):
    """Provider for OpenShift OC CLI tools."""

    @property
    def tools(self) -> dict[str, BaseTool]:
        """Get all OC tools."""
        return {
            t.name: t
            for t in [
                oc_get,
                oc_describe,
                oc_logs,
                oc_adm_top,
                oc_status,
                show_pods,
            ]
        }

    # TODO: needs rebase for #2391
    def execute_tool(
        self, tool_name: str, tool_args: dict, context: ToolsContext
    ) -> tuple[str, str]:
        """Execute an OC tool with the given arguments and context."""
        tool = self.tools[tool_name]
        if not context.user_token:
            return "Error: No user token provided", "error"

        # add token to arguments
        args_with_token = {**tool_args, "token": context.user_token}

        try:
            return tool.invoke(args_with_token), "success"
        except Exception as e:
            return f"Error: {e}", "error"
