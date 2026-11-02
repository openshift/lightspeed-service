"""Utilities for interacting with the OpenShift cluster."""

import base64
import json
import os
from uuid import uuid4

import pytest
from openshift.dynamic.exceptions import ConflictError, ForbiddenError, NotFoundError

from tests.e2e.utils import cluster_api
from tests.e2e.utils.openshift_client import REQUEST_TIMEOUT
from tests.e2e.utils.retry import retry_until_timeout_or_success
from tests.e2e.utils.wait_for_ols import wait_for_ols

POD_READY_RETRY_COUNT = 120
_SESSION_TOKEN_SECRETS: set[str] = set()


class PodNotFoundError(Exception):
    """Raised when a pod is not found on the cluster."""


def get_cluster_id() -> str:
    """Get the cluster ID."""
    version = cluster_api.read_resource("clusterversion", "version")
    return version["spec"]["clusterID"]


def get_cluster_version() -> tuple[str, str]:
    """Get the cluster version: major and minor."""
    version = cluster_api.read_resource("clusterversion", "version")
    major, minor, _ = version["status"]["desired"]["version"].split(".", 2)
    return major, minor


def deployment_exists(name: str, namespace: str = "openshift-lightspeed") -> bool:
    """Return whether a Deployment exists, propagating authentication failures."""
    resource = cluster_api.resource_for("deployment")
    try:
        resource.get(name=name, namespace=namespace, _request_timeout=REQUEST_TIMEOUT)
    except NotFoundError:
        return False
    return True


def lightspeed_operator_manager_deployed(
    namespace: str = "openshift-lightspeed",
) -> bool:
    """Return True if the operator controller Deployment exists.

    Used to detect a pre-installed operator without OLM (e.g. Konflux direct /
    ``make deploy``). Bundle installs also create this Deployment, but those are
    usually detected via ``ClusterServiceVersion`` first.
    """
    return deployment_exists("lightspeed-operator-controller-manager", namespace)


def create_user(name: str, ignore_existing_resource: bool = False) -> None:
    """Create a service account user for testing."""
    resource = cluster_api.resource_for("serviceaccount")
    try:
        resource.get(
            name=name,
            namespace="openshift-lightspeed",
            _request_timeout=REQUEST_TIMEOUT,
        )
        return
    except NotFoundError:
        pass
    try:
        resource.create(
            namespace="openshift-lightspeed",
            body={
                "apiVersion": "v1",
                "kind": "ServiceAccount",
                "metadata": {"name": name},
            },
            _request_timeout=REQUEST_TIMEOUT,
        )
    except ConflictError:
        if not ignore_existing_resource:
            raise


def delete_user(name: str) -> None:
    """Delete a service account user."""
    cluster_api.resource_for("serviceaccount").delete(
        name=name, namespace="openshift-lightspeed", _request_timeout=REQUEST_TIMEOUT
    )


def get_token_for(name: str) -> str:
    """Create a session token secret, preserving the former CLI token lifetime.

    Evaluation subprocesses reuse the token for the entire suite and cannot
    refresh TokenRequest credentials. Session cleanup removes the token secrets.
    """
    secret_name = f"{name}-token-{uuid4().hex[:12]}"
    _SESSION_TOKEN_SECRETS.add(secret_name)
    cluster_api.create_resource(
        {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": secret_name,
                "annotations": {"kubernetes.io/service-account.name": name},
            },
            "type": "kubernetes.io/service-account-token",
        }
    )
    token = ""
    read_error: ForbiddenError | None = None

    def populated() -> bool:
        nonlocal token, read_error
        try:
            secret = cluster_api.read_resource("secret", secret_name)
        except ForbiddenError as error:
            read_error = error
            return True
        token = secret.get("data", {}).get("token", "")
        return bool(token)

    populated_in_time = retry_until_timeout_or_success(
        30, 1, populated, "Waiting for service account token"
    )
    if read_error is not None:
        raise read_error
    if not populated_in_time:
        raise TimeoutError(f"Token was not populated for service account {name}")
    return base64.b64decode(token).decode("utf-8")


def cleanup_session_tokens() -> None:
    """Delete every token created by this session, including partially failed setup."""
    errors = []
    for name in tuple(_SESSION_TOKEN_SECRETS):
        try:
            cluster_api.delete_resource("secret", name, ignore_missing=True, timeout=60)
        except Exception as error:
            errors.append(error)
        else:
            _SESSION_TOKEN_SECRETS.remove(name)
    if errors:
        raise ExceptionGroup("Failed to clean up session token secrets", errors)


def grant_sa_user_access(name: str, role: str) -> None:
    """Grant access with a binding garbage-collected alongside the test namespace."""
    namespace = cluster_api.read_resource("namespace", "openshift-lightspeed")
    cluster_api.apply_resource(
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {
                "name": f"ols-e2e-{name}-{role}",
                "ownerReferences": [
                    {
                        "apiVersion": "v1",
                        "kind": "Namespace",
                        "name": "openshift-lightspeed",
                        "uid": namespace["metadata"]["uid"],
                    }
                ],
            },
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": role,
            },
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": name,
                    "namespace": "openshift-lightspeed",
                }
            ],
        }
    )


def get_ols_url(route_name: str) -> str:
    """Get the URL for the OLS route."""
    route = cluster_api.resource_for("route").get(
        name=route_name,
        namespace="openshift-lightspeed",
        _request_timeout=REQUEST_TIMEOUT,
    )
    return "https://" + route.spec.host


def get_running_pods(namespace: str = "openshift-lightspeed") -> list[str]:
    """Get the names of all running pods in the cluster."""
    pods = cluster_api.resource_for("pod").get(
        namespace=namespace,
        field_selector="status.phase=Running",
        _request_timeout=REQUEST_TIMEOUT,
    )
    return [pod.metadata.name for pod in pods.items]


def get_pod_by_prefix(
    prefix: str = "lightspeed-app-server-",
    namespace: str = "openshift-lightspeed",
    fail_not_found: bool = True,
) -> list[str]:
    """Return name of the running pod(s) which match the specified prefix."""
    result = get_running_pods(namespace)
    pods = [pod for pod in result if prefix in pod]
    if fail_not_found and not pods:
        assert False, f"No OLS api server pod found in list pods: {result}"
    return pods


def get_pod_containers(pod: str, namespace: str = "openshift-lightspeed") -> list[str]:
    """Get the names of all containers in the pod."""
    resource = cluster_api.resource_for("pod")
    try:
        result = resource.get(
            name=pod, namespace=namespace, _request_timeout=REQUEST_TIMEOUT
        )
    except NotFoundError as e:
        raise PodNotFoundError(f"Pod {pod} not found") from e
    return [container.name for container in result.spec.containers]


def list_path(pod_name: str, path: str) -> list[str]:
    """List the contents of a path in a pod, returning an empty list when absent."""
    result = cluster_api.execute(
        pod_name,
        [
            "sh",
            "-c",
            'if [ -e "$1" ] || [ -L "$1" ]; then ls "$1"; fi',
            "--",
            path,
        ],
    )
    return result.splitlines()


def remove_dir(pod_name: str, directory: str) -> None:
    """Remove a directory in a pod."""
    cluster_api.execute(pod_name, ["rm", "-rf", "--", directory])


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

    return json.loads(cluster_api.execute(pod_name, ["cat", "--", full_path]))


def get_single_existing_feedback(
    pod_name: str, feedbacks_path: str, retries: int = 5, delay: float = 2.0
) -> dict:
    """Return the content of the single feedback that is in the cluster.

    Retry the filesystem read to allow for network/PVC propagation delay
    after the API returns 200.
    """
    feedbacks: list = []

    def _check() -> bool:
        nonlocal feedbacks
        feedbacks = list_path(pod_name, feedbacks_path) or []
        return len(feedbacks) == 1

    success = retry_until_timeout_or_success(
        retries, delay, _check, description="waiting for feedback file"
    )
    assert success, (
        f"Expected exactly 1 feedback file in {feedbacks_path} after "
        f"{retries} retries ({retries * delay}s), got {len(feedbacks)}: {feedbacks}"
    )

    full_path = f"{feedbacks_path}/{feedbacks[0]}"

    return json.loads(cluster_api.execute(pod_name, ["cat", "--", full_path]))


def get_container_log(pod_name: str, container_name: str) -> str:
    """Return the logs of a container in a pod."""
    return cluster_api.pod_logs(pod_name, container_name)


def create_file(pod_name: str, path: str, content: str) -> None:
    """Create a file in a pod."""
    dir_path = path.rsplit("/", 1)[0]
    cluster_api.execute(pod_name, ["mkdir", "-p", "--", dir_path])
    cluster_api.execute(
        pod_name, ["sh", "-c", 'printf "%s\\n" "$1" > "$2"', "--", content, path]
    )


def remove_file(pod_name: str, path: str) -> None:
    """Remove a file in a pod."""
    cluster_api.execute(pod_name, ["rm", "--", path])


def get_container_ready_status(
    pod: str, namespace: str = "openshift-lightspeed"
) -> list[str]:
    """Get status for all containers in pod."""
    result = cluster_api.resource_for("pod").get(
        name=pod, namespace=namespace, _request_timeout=REQUEST_TIMEOUT
    )
    statuses = result.status.containerStatuses or []
    return [str(status.ready).lower() for status in statuses]


def wait_for_running_pod(
    name: str = "lightspeed-app-server-",
    namespace: str = "openshift-lightspeed",
    wait_http_ready: bool = True,
):
    """Wait for the selected pod to be in running state.

    After the pod's containers are ready, optionally poll ``/readiness`` on the
    OLS route so callers do not hit ingress/HTML error pages before JSON is served
    again (e.g. after scale + config changes).
    """
    r = retry_until_timeout_or_success(
        POD_READY_RETRY_COUNT,
        3,
        lambda: len(
            get_pod_by_prefix(prefix=name, namespace=namespace, fail_not_found=False)
        )
        == 1,
        "Waiting for service pod in running state",
    )
    if not r:
        raise Exception("Timed out waiting for new OLS pod to be ready")

    def pod_has_containers_ready():
        pods = get_pod_by_prefix(prefix=name, namespace=namespace, fail_not_found=False)
        if not pods:
            return False

        statuses = get_container_ready_status(pods[0], namespace)
        return len(statuses) > 0 and all(s == "true" for s in statuses)

    r = retry_until_timeout_or_success(
        POD_READY_RETRY_COUNT,
        5,
        pod_has_containers_ready,
        "Waiting for containers in the server pod to become ready",
    )
    if not r:
        raise Exception("Timed out waiting for containers to become ready")

    if not wait_http_ready:
        return

    ols_url_env = os.getenv("OLS_URL", "")
    if "localhost" in ols_url_env:
        return

    http_url = getattr(pytest, "ols_url", None) or ""
    if not http_url.strip():
        http_url = get_ols_url("ols")

    if not wait_for_ols(http_url, timeout=360, interval=5):
        raise RuntimeError(
            "Timed out waiting for OLS HTTP readiness after pod became ready"
        )


def get_certificate_secret_name(
    name: str = "lightspeed-app-server", namespace: str = "openshift-lightspeed"
) -> str:
    """Get the name of the certificates secret for the service."""
    service = cluster_api.resource_for("service").get(
        name=name, namespace=namespace, _request_timeout=REQUEST_TIMEOUT
    )
    return service.metadata.annotations[
        "service.beta.openshift.io/serving-cert-secret-name"
    ]


def delete_resource(resource: str, name: str, namespace: str = "") -> None:
    """Delete the selected resource and wait for deletion to complete."""
    cluster_api.delete_resource(resource, name, namespace or cluster_api.NAMESPACE)


def restart_deployment(name: str, namespace: str) -> None:
    """Restart the selected deployment."""
    for replicas in (0, 1):
        cluster_api.scale_deployment(name, replicas, namespace)
