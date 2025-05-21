"""OpenShift CLI tools."""

import logging
import os
import subprocess
import traceback

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("oc_cli_tools")


BLOCKED_CHARS = (";", "&", "|", "`", "$", "(", ")", "<", ">", "\\")
BLOCKED_CHARS_DETECTED_MSG = (
    f"Error: arguments contain blocked characters: {BLOCKED_CHARS}"
)
SECRET_NOT_ALLOWED_MSG = (
    "Error: 'secret' or 'secrets' are not allowed in arguments"  # noqa: S105
)


def strip_args_for_oc_command(args: list[str]) -> list[str]:
    """Sanitize arguments for `oc` CLI commands if LLM adds extras."""
    # fixes these cases:
    # - extra spaces in args: ["pod "]
    # - extra commands in args: ["oc", "get", "pod"]
    # - two commands as one in args: ["pod my-pod"]
    remove_args = {"oc", "get", "describe", "logs", "status", "adm", "top"}
    split_args = [
        arg for arg in " ".join(args).split() if arg and arg not in remove_args
    ]
    return split_args


def is_blocked_char_in_args(args: list[str]) -> bool:
    """Check if any of the arguments contain blocked characters."""
    arg_str = " ".join(args)
    if any(char in arg_str for char in BLOCKED_CHARS):
        logger.error("Blocked characters found in argument: %s", arg_str)
        return True
    return False


def is_secret_in_args(args: list[str]) -> bool:
    """Check if any of the arguments contain 'secret'."""
    arg_str = " ".join(args)
    if "secret" in arg_str:
        logger.error("'secret' keyword found in argument: %s", arg_str)
        return True
    return False


def redact_token(text: str, token: str) -> str:
    """Redact token from text."""
    return text.replace(token, "<redacted>")


def resolve_response(result: subprocess.CompletedProcess) -> str:
    """Return stdout if it is not empty string, otherwise return stderr."""
    # some commands returns empty stdout and message like "namespace not found"
    # in stderr, but with return code 0
    if result.returncode == 0:
        return result.stdout if result.stdout != "" else result.stderr
    return result.stderr


def run_oc(args: list[str]) -> str:
    """Run `oc` CLI with provided arguments and command."""
    # Currently user token is sent to server using env var.
    token = os.environ.get("OC_USER_TOKEN", "token-not-set")

    try:
        res = subprocess.run(  # noqa: S603
            ["oc", *args, "--token", token],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
    except Exception:
        # if token was used, redact the error to ensure it is not leaked
        return f"Error executing args '{args}': {redact_token(traceback.format_exc(), token)}"

    response = resolve_response(res)
    return redact_token(response, token)


def safe_run_oc(commands: list[str], args: list[str]) -> str:
    """Run `oc` CLI with provided arguments and command."""
    if is_blocked_char_in_args(args):
        return BLOCKED_CHARS_DETECTED_MSG
    if is_secret_in_args(args):
        return SECRET_NOT_ALLOWED_MSG
    result = run_oc([*commands, *strip_args_for_oc_command(args)])
    return result


@mcp.tool()
def oc_get(oc_get_args: list[str]) -> str:
    """Display one or many resources from OpenShift cluster.

    Standard `oc` flags and options are valid.

    Namespace is optional argument. If not provided, the default namespace will be used.
    To specify a namespace, use the `--namespace <namespace>` or `-n <namespace>`.

    Args:
        oc_get_args: Arguments for oc get

    Examples:
        # List all pods in ps output format.
        oc get pods

        # List all pods in ps output format with more information (such as node name).
        oc get pods -o wide

        # List events for cluster
        oc get events

        # List a single replication controller with specified NAME in ps output format.
        oc get replicationcontroller web

        # List deployments in JSON output format, in the "v1" version of the "apps" API group:
        oc get deployments.v1.apps -o json

        # List a pod identified by type and name specified in "pod.yaml" in JSON output format.
        oc get -f pod.yaml -o json

        # List all replication controllers and services together in ps output format.
        oc get rc,services
    """
    return safe_run_oc(["get"], oc_get_args)


@mcp.tool()
def oc_describe(oc_describe_args: list[str]) -> str:
    """Show details of a specific resource or group of resources.

    Print a detailed description of the selected resources, including related resources such as events or controllers.
    You may select a single object by name, all objects of that type, provide a name prefix, or label selector.

    Namespace is optional argument. If not provided, the default namespace will be used.
    To specify a namespace, use the `--namespace <namespace>` or `-n <namespace>`.

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
    return safe_run_oc(["describe"], oc_describe_args)


@mcp.tool()
def oc_logs(oc_logs_args: list[str]) -> str:
    """Print the logs for a resource.

    Supported resources are builds, build configs (bc), deployment configs (dc), and pods.
    When a pod is specified and has more than one container, the container name should be specified via -c.
    When a build config or deployment config is specified, you can view the logs for a particular version of it via --version.

    Namespace is optional argument. If not provided, the default namespace will be used.
    To specify a namespace, use the `--namespace <namespace>` or `-n <namespace>`.

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
    return safe_run_oc(["logs"], oc_logs_args)


@mcp.tool()
def oc_status(oc_status_args: list[str]) -> str:
    """Show a high level overview of the current project.

    This command will show services, deployment configs, build configurations, & active deployments.
    If you have any misconfigured components information about them will be shown.
    For more information about individual items, use the describe command \
    (e.g. oc describe buildconfig, oc describe deploymentconfig, oc describe service).

    Namespace is optional argument. If not provided, the default namespace will be used.
    To specify a namespace, use the `--namespace <namespace>` or `-n <namespace>`.

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
    return safe_run_oc(["status"], oc_status_args)


@mcp.tool()
def show_pods_resource_usage() -> str:
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
    return run_oc(["adm", "top", "pods", "-A"])


@mcp.tool()
def oc_adm_top(oc_adm_top_args: list[str]) -> str:
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
    return safe_run_oc(["adm", "top"], oc_adm_top_args)


if __name__ == "__main__":
    mcp.run(transport="stdio")
