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

logger = logging.getLogger(__name__)


BLOCKED_CHARS = (";", "&", "|", "`", "$", "(", ")", "<", ">", "\\")
BLOCKED_CHARS_DETECTED_MSG = (
    f"Error: arguments contain blocked characters: {BLOCKED_CHARS}"
)
SECRET_NOT_ALLOWED_MSG = (
    "Error: 'secret' or 'secrets' are not allowed in arguments"  # noqa: S105
)


def strip_args_for_oc_command(args: list[str]) -> list[str]:
    """Sanitize arguments for `oc` CLI commands if LLM adds it extra."""
    # Sometimes within the list we may get two args combined; ex: [top pod]
    striped_args = " ".join(args).split(" ")
    # Sometimes model gives args which are already added to the tool.
    remove_arg = ["oc", "get", "describe", "logs", "status", "adm", "top"]
    for arg in remove_arg:
        if arg in striped_args:
            logger.debug("Removing argument from sanitized args: %s", arg)
            striped_args.remove(arg)

    return striped_args


def is_blocked_char_in_args(args: list[str]) -> list[str]:
    """Check if any of the arguments contain blocked characters."""
    arg_str = " ".join(args)
    if any(char in arg_str for char in BLOCKED_CHARS):
        logger.error("Blocked characters found in argument: %s", arg_str)
        return True
    return False


def is_secret_in_args(args: list[str]) -> list[str]:
    """Check if any of the arguments contain 'secret' or 'secrets'."""
    for arg in args:
        if "secret" in arg or "secrets" in arg:
            logger.error(
                "'secret' or 'secrets' found in argument: %s",
                arg,
            )
            return True
    return False


def stdout_or_stderr(result: subprocess.CompletedProcess) -> str:
    """Return stdout if it is not empty string, otherwise return stderr."""
    # some commands returns empty stdout and message like "namespace not found"
    # in stderr, but with return code 0
    return result.stdout if result.stdout != "" else result.stderr


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


def safe_run_oc(args: list[str], token: str) -> str:
    """Run `oc` CLI with provided arguments and command."""
    if is_blocked_char_in_args(args):
        return BLOCKED_CHARS_DETECTED_MSG
    if is_secret_in_args(args):
        return SECRET_NOT_ALLOWED_MSG

    result = run_oc(["get", *strip_args_for_oc_command(args), "--token", token])
    return stdout_or_stderr(result)


def token_works_for_oc(token: str) -> bool:
    """Check if the token can be used with `oc` CLI.

    Args:
        token: OpenShift user token.

    Returns:
        True if user token works, False otherwise.
    """
    try:
        run_oc(["version", f"--token={token}"])
        logger.info("Token is usable for oc CLI")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(
            "Unable to use the token for oc CLI; stdout: %s, stderr: %s",
            e.stdout,
            e.stderr,
        )
        return False


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
    return safe_run_oc(["get", *oc_get_args], token)


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
    return safe_run_oc(["describe", *oc_describe_args], token)


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
    return safe_run_oc(["logs", *oc_logs_args], token)


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
    return safe_run_oc(["status", *oc_status_args], token)


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
    return safe_run_oc(["adm", "top", *oc_adm_top_args], token)
