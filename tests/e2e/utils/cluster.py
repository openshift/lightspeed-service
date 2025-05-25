"""Utilities for interacting with the OpenShift cluster."""

import json
import subprocess

from tests.e2e.utils.retry import retry_until_timeout_or_success

OC_COMMAND_RETRY_COUNT = 120


def run_oc(
    args: list[str], command=None, ignore_existing_resource=False
) -> subprocess.CompletedProcess:
    """Run a command in the OpenShift cluster."""
    try:
        res = subprocess.run(  # noqa: S603
            ["oc", *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            input=command,
        )
        return res
    except subprocess.CalledProcessError as e:
        if ignore_existing_resource and "AlreadyExists" in e.stderr:
            print(f"Resource already exists: {e}\nproceeding...")
        else:
            print(
                f"Error running oc command {args}: {e}, stdout: {e.output}, stderr: {e.stderr}"
            )
            raise
    return subprocess.CompletedProcess("", 0)


def run_oc_and_store_stdout(
    args: list[str], stdout_file: str
) -> subprocess.CompletedProcess:
    """Run a command in the OpenShift cluster and store the stdout in a file."""
    try:
        result = run_oc(args)
        assert result.returncode == 0
        with open(stdout_file, "w", encoding="utf-8") as fout:
            fout.write(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        raise Exception("Error running oc command") from e


def get_cluster_id() -> str:
    """Get the cluster ID."""
    try:
        result = run_oc(
            [
                "get",
                "clusterversions",
                "version",
                "-o",
                "jsonpath='{.spec.clusterID}'",
            ]
        )
        return result.stdout.strip("'")
    except subprocess.CalledProcessError as e:
        raise Exception("Error getting cluster ID") from e


def get_cluster_version() -> tuple[str, str]:
    """Get the cluster version: major and minor."""
    try:
        result = run_oc(
            [
                "get",
                "clusterversions",
                "version",
                "-o",
                "jsonpath='{.status.desired.version}'",
            ]
        )
        major, minor, _ = result.stdout.strip("'").split(".", 2)
        return major, minor
    except subprocess.CalledProcessError as e:
        raise Exception("Error getting cluster version") from e


def create_user(name: str, ignore_existing_resource=False) -> None:
    """Create a service account user for testing."""
    try:
        try:
            run_oc(["get", "sa", name])
            print("Service account %s already exists", name)
        except subprocess.CalledProcessError:
            run_oc(
                ["create", "sa", name],
                ignore_existing_resource=ignore_existing_resource,
            )
    except subprocess.CalledProcessError as e:
        raise Exception("Error creating service account") from e


def delete_user(name: str) -> None:
    """Delete a service account user."""
    try:
        run_oc(["delete", "sa", name])
    except subprocess.CalledProcessError as e:
        raise Exception("Error deleting service account") from e


def get_token_for(name: str) -> str:
    """Get the token for the service account user."""
    try:
        result = run_oc(["sa", "new-token", name])
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise Exception("Error getting token for service account") from e


def grant_sa_user_access(name: str, role: str) -> None:
    """Grant the service account user access to OLS."""
    try:
        run_oc(
            [
                "adm",
                "policy",
                "add-cluster-role-to-user",
                role,
                "-z",
                name,
            ]
        )
    except subprocess.CalledProcessError as e:
        raise Exception("Error granting user access") from e


def get_ols_url(route_name: str) -> str:
    """Get the URL for the OLS route."""
    try:
        result = run_oc(
            [
                "get",
                "route",
                route_name,
                "-o",
                "jsonpath={.spec.host}",
            ]
        )
        hostname = result.stdout.strip()
        return "https://" + hostname
    except subprocess.CalledProcessError as e:
        raise Exception("Error getting route hostname") from e


def get_running_pods(namespace: str = "openshift-lightspeed") -> list[str]:
    """Get the names of all running pods in the cluster."""
    try:
        result = run_oc(
            [
                "get",
                "pods",
                "--field-selector=status.phase=Running",
                "-n",
                namespace,
                "-o",
                "jsonpath='{.items[*].metadata.name}'",
            ]
        )
        return result.stdout.strip("'").split()
    except subprocess.CalledProcessError as e:
        raise Exception("Error getting pods") from e


def get_pod_by_prefix(
    prefix: str = "lightspeed-app-server-",
    namespace: str = "openshift-lightspeed",
    fail_not_found: bool = True,
) -> list[str]:
    """Return name of the running pod(s) which match the specified prefix."""
    pods = []
    try:
        result = get_running_pods(namespace)
        pods = [pod for pod in result if prefix in pod]
        if fail_not_found and not pods:
            assert False, f"No OLS api server pod found in list pods: {result}"
        return pods
    except subprocess.CalledProcessError as e:
        raise Exception("Error getting pod name") from e


def get_pod_containers(pod, namespace: str = "openshift-lightspeed") -> list[str]:
    """Get the names of all containers in the pod."""
    try:
        result = run_oc(
            [
                "get",
                "-n",
                namespace,
                "pod",
                pod,
                "-o",
                "jsonpath='{.spec.containers[*].name}'",
            ]
        )
        return result.stdout.strip("'").split()
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error getting containers of pod {pod}") from e


def list_path(pod_name: str, path: str) -> list[str]:
    """List the contents of a path in a pod."""
    try:
        result = run_oc(
            [
                "rsh",
                pod_name,
                "ls",
                path,
            ]
        )
        # files are returned as 'file1\nfile2\n'
        return [f for f in result.stdout.split("\n") if f]
    except subprocess.CalledProcessError as e:
        print(f"Error listing path {path}: {e}, stderr: {e.stderr}, stdout: {e.stdout}")
        if e.returncode == 2 and (
            "No such file or directory" in e.stdout
            or "No such file or directory" in e.stderr
        ):
            return []
        raise Exception("Error listing pod path") from e


def remove_dir(pod_name: str, directory: str) -> None:
    """Remove a directory in a pod."""
    try:
        run_oc(["rsh", pod_name, "rm", "-rf", directory])
    except subprocess.CalledProcessError as e:
        raise Exception("Error removing directory") from e


def get_single_existing_transcript(pod_name: str, transcripts_path: str) -> dict:
    """Return the content of the single transcript that is in the cluster."""
    user_id_list = list_path(pod_name, transcripts_path)
    assert len(user_id_list) == 1
    user_id = user_id_list[0]
    conv_id_list = list_path(pod_name, transcripts_path + "/" + user_id)
    assert len(conv_id_list) == 1
    conv_id = conv_id_list[0]
    transcript_list = list_path(
        pod_name, transcripts_path + "/" + user_id + "/" + conv_id
    )
    assert len(transcript_list) == 1
    transcript = transcript_list[0]

    full_path = f"{transcripts_path}/{user_id}/{conv_id}/{transcript}"

    try:
        transcript_content = run_oc(["exec", pod_name, "--", "cat", full_path])
        return json.loads(transcript_content.stdout)
    except subprocess.CalledProcessError as e:
        raise Exception("Error reading transcript") from e


def get_single_existing_feedback(pod_name: str, feedbacks_path: str) -> dict:
    """Return the content of the single feedback that is in the cluster."""
    feedbacks = list_path(pod_name, feedbacks_path)
    assert len(feedbacks) == 1
    feedback = feedbacks[0]

    full_path = f"{feedbacks_path}/{feedback}"

    try:
        feedback_content = run_oc(["exec", pod_name, "--", "cat", full_path])
        return json.loads(feedback_content.stdout)
    except subprocess.CalledProcessError as e:
        raise Exception("Error reading feedback") from e


def get_container_log(pod_name: str, container_name: str) -> str:
    """Return the logs of a container in a pod."""
    try:
        result = run_oc(["logs", "--follow=false", pod_name, "-c", container_name])
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise Exception("Error getting container logs") from e


def create_file(pod_name: str, path: str, content: str) -> None:
    """Create a file in a pod."""
    try:
        # ensure dir exists
        dir_path = path.rsplit("/", 1)[0]  # without file
        run_oc(["exec", pod_name, "--", "mkdir", "-p", dir_path])
        run_oc(["exec", pod_name, "--", "sh", "-c", f"echo '{content}' > {path}"])
    except subprocess.CalledProcessError as e:
        raise Exception("Error creating file") from e


def remove_file(pod_name: str, path: str) -> None:
    """Remove a file in a pod."""
    try:
        run_oc(["exec", pod_name, "--", "rm", path])
    except subprocess.CalledProcessError as e:
        raise Exception("Error removing file") from e


def get_container_ready_status(pod: str, namespace: str = "openshift-lightspeed"):
    """Get status for all containers in pod."""
    try:
        result = run_oc(
            [
                "get",
                "-n",
                namespace,
                "pod",
                pod,
                "-o",
                "jsonpath='{.status.containerStatuses[*].ready}'",
            ]
        )
        return result.stdout.strip("'").split()
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error getting containers of pod {pod}") from e


def wait_for_running_pod(
    name: str = "lightspeed-app-server-", namespace: str = "openshift-lightspeed"
):
    """Wait for the selected pod to be in running state."""
    r = retry_until_timeout_or_success(
        5,
        3,
        lambda: len(
            run_oc(
                [
                    "get",
                    "pods",
                    "--field-selector=status.phase=Pending",
                    "-n",
                    namespace,
                ]
            ).stdout
        )
        == 1,
        "Just one pod is in pending state",
    )

    # wait for new ols app pod to be created+running
    # there should be exactly one, if we see more than one it may be an old pod
    # and we need to wait for it to go away before progressing so we don't try to
    # interact with it.
    r = retry_until_timeout_or_success(
        OC_COMMAND_RETRY_COUNT,
        5,
        lambda: len(
            get_pod_by_prefix(prefix=name, namespace=namespace, fail_not_found=False)
        )
        == 1,
        "Waiting for service pod in running state",
    )
    if not r:
        raise Exception("Timed out waiting for new OLS pod to be ready")

    def pod_has_2_containers_ready():
        pods = get_pod_by_prefix(prefix=name, namespace=namespace, fail_not_found=False)
        if not pods:
            return False
        return (
            len(
                [
                    container
                    for container in get_container_ready_status(pods[0])
                    if container == "true"
                ]
            )
            == 2
        )

    # wait for the two containers in the server pod to become ready
    r = retry_until_timeout_or_success(
        OC_COMMAND_RETRY_COUNT,
        5,
        pod_has_2_containers_ready,
        "Waiting for two containers in the server pod to become ready",
    )
    if not r:
        raise Exception("Timed out waiting for new two containers to become ready")


def get_certificate_secret_name(
    name: str = "lightspeed-app-server", namespace: str = "openshift-lightspeed"
) -> str:
    """Get the name of the certificates secret for the service."""
    try:
        result = run_oc(
            [
                "get",
                "service",
                name,
                "-n",
                namespace,
                "-o",
                "json",
            ]
        )
        return json.loads(result.stdout)["metadata"]["annotations"][
            "service.beta.openshift.io/serving-cert-secret-name"
        ]
    except subprocess.CalledProcessError as e:
        raise Exception("Error getting serving cert secret name") from e


def delete_resource(resource: str, name: str, namespace: str = "") -> None:
    """Delete the selected resource. If namespace is None, we assume it's a cluster resource."""
    if namespace:
        try:
            run_oc(
                [
                    "delete",
                    resource,
                    name,
                    "-n",
                    namespace,
                ]
            )
            return
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error deleting the {name} instance of {resource}") from e
    try:
        run_oc(
            [
                "delete",
                resource,
                name,
            ]
        )
        return
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error deleting the {name} instance of {resource}") from e


def restart_deployment(name: str, namespace: str) -> None:
    """Restart the selected deployment."""
    try:
        run_oc(
            [
                "scale",
                f"deployment/{name}",
                "--replicas",
                "0",
                "-n",
                namespace,
            ]
        )
    except subprocess.CalledProcessError as e:
        raise Exception("Error scaling replicas to 0") from e
    try:
        run_oc(
            [
                "scale",
                f"deployment/{name}",
                "--replicas",
                "1",
                "-n",
                namespace,
            ]
        )
    except subprocess.CalledProcessError as e:
        raise Exception("Error scaling replicas to 1") from e
