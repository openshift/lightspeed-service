"""OpenShift CLI tools."""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def sanitize_oc_args(args: list[str]) -> list[str]:
    """Sanitize `oc` CLI arguments."""
    # TODO: check for malicious arguments, eg. `| rm -rf /`
    return args


def run_oc(args: list[str]) -> subprocess.CompletedProcess:
    """Run `oc` CLI with provided arguments and command."""
    try:
        res = subprocess.run(  # noqa: S603
            ["oc", *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        return res
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Error running oc command {args}: {e}, stdout: {e.output}, stderr: {e.stderr}"  # noqa: G004
        )
        raise


def log_to_oc(token: str, server: str = os.getenv("KUBERNETES_SERVICE_HOST")) -> bool:
    """Log in to `oc` CLI via provided token.

    Args:
        token: OpenShift token.
        server: OpenShift server address.

    Returns:
        True if login was successful, False otherwise.
    """
    if not server:
        logger.error(
            "KUBERNETES_SERVICE_HOST env is not set, please provide"
            "server address to log in to OpenShift"
        )
        return False
    try:
        run_oc(
            [
                "login",
                server,
                "--token",
                token,
                # TODO: remove tls skip when possible, but it might hold the
                # prompt for user input "y/n" without it
                # So verify certs first?
                "--insecure-skip-tls-verify=true",
            ],
        )
        logger.info("Successfully logged in to OpenShift")
        return True
    except subprocess.CalledProcessError as e:
        # TODO: not ideal - the message can change in future
        if any(substring in e.stderr for substring in ("token", "invalid", "expired")):
            logger.error("Error logging in to OpenShift: token is invalid or expired")
            return False
        else:
            raise e


def oc_get(args: list[str]) -> subprocess.CompletedProcess:
    """Display one or many resources from OpenShift cluster using `oc get <args>` command.

    Examples:
        # List all pods in ps output format.
        oc get pods

        # List all pods in ps output format with more information (such as node name).
        oc get pods -o wide

        # List a single replication controller with specified NAME in ps output format.
        oc get replicationcontroller web

        # List deployments in JSON output format, in the "v1" version of the "apps" API group:
        oc get deployments.v1.apps -o json

        # List a single pod in JSON output format.
        oc get -o json pod web-pod-13je7

        # List a pod identified by type and name specified in "pod.yaml" in JSON output format.
        oc get -f pod.yaml -o json

        # List resources from a directory with kustomization.yaml - e.g. dir/kustomization.yaml.
        oc get -k dir/

        # Return only the phase value of the specified pod.
        oc get -o template pod/web-pod-13je7 --template={{.status.phase}}

        # List resource information in custom columns.
        oc get pod test-pod -o
        custom-columns=CONTAINER:.spec.containers[0].name,IMAGE:.spec.containers[0].image

        # List all replication controllers and services together in ps output format.
        oc get rc,services

        # List one or more resources by their type and names.
        oc get rc/web service/frontend pods/web-pod-13je7
    """
    result = run_oc(["get", *sanitize_oc_args(args)])
    return result.stdout


# TODO: better description - copy from `oc <command> --help`?
def oc_describe(args: list[str]) -> subprocess.CompletedProcess:
    """Describe resource from OpenShift cluster."""
    result = run_oc(["describe", *sanitize_oc_args(args)])
    return result.stdout


def oc_logs(args: list[str]) -> subprocess.CompletedProcess:
    """Get logs from OpenShift cluster."""
    result = run_oc(["logs", *sanitize_oc_args(args)])
    return result.stdout


def oc_status(args: list[str]) -> subprocess.CompletedProcess:
    """Get status of resource from OpenShift cluster."""
    result = run_oc(["status", *sanitize_oc_args(args)])
    return result.stdout


def oc_adm_top(args: list[str]) -> subprocess.CompletedProcess:
    """Get top resources from OpenShift cluster."""
    result = run_oc(["adm", "top", *sanitize_oc_args(args)])
    return result.stdout
