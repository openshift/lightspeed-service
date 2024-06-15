"""Utilities for interacting with the OpenShift cluster."""

import json
import subprocess


def run_oc(args: list[str]) -> subprocess.CompletedProcess:
    """Run a command in the OpenShift cluster."""
    return subprocess.run(
        ["oc", *args],  # noqa: S603, S607
        capture_output=True,
        text=True,
        check=True,
    )


def run_oc_and_store_stdout(
    args: list[str], stdout_file: str
) -> subprocess.CompletedProcess:
    """Run a command in the OpenShift cluster and store the stdout in a file."""
    try:
        result = run_oc(args)
        assert result.returncode == 0
        with open(stdout_file, "w") as fout:
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
        major, minor, rest = result.stdout.strip("'").split(".", 2)
        return major, minor
    except subprocess.CalledProcessError as e:
        raise Exception("Error getting cluster version") from e


def create_user(name: str) -> None:
    """Create a service account user for testing."""
    try:
        run_oc(["create", "sa", name])
    except subprocess.CalledProcessError as e:
        raise Exception("Error creating service account") from e


def delete_user(name: str) -> None:
    """Delete a service account user."""
    try:
        run_oc(["delete", "sa", name])
    except subprocess.CalledProcessError as e:
        raise Exception("Error deleting service account") from e


def get_user_token(name: str) -> str:
    """Get the token for the service account user."""
    try:
        result = run_oc(["create", "token", name])
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


def get_pods(namespace: str = "openshift-lightspeed") -> list[str]:
    """Get the names of all pods in the cluster."""
    try:
        result = run_oc(
            [
                "get",
                "pods",
                "-n",
                namespace,
                "-o",
                "jsonpath='{.items[*].metadata.name}'",
            ]
        )
        return result.stdout.strip("'").split()
    except subprocess.CalledProcessError as e:
        raise Exception("Error getting pods") from e


def get_single_existing_pod_name(namespace: str = "openshift-lightspeed") -> str:
    """Return name of the single pod that is in the cluster."""
    try:
        result = get_pods(namespace)
        assert len(result) == 1
        return result[0]
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
        if e.returncode == 2 and "No such file or directory" in e.stderr:
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
