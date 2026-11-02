"""Access cluster resources directly for test setup and diagnostics."""

import base64
import time
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from kubernetes import client
from kubernetes.stream import stream
from openshift.dynamic.exceptions import ConflictError, NotFoundError

from tests.e2e.utils.openshift_client import REQUEST_TIMEOUT, get_client

NAMESPACE = "openshift-lightspeed"
RESOURCES = {
    "namespace": ("v1", "Namespace"),
    "clusterversion": ("config.openshift.io/v1", "ClusterVersion"),
    "pod": ("v1", "Pod"),
    "service": ("v1", "Service"),
    "configmap": ("v1", "ConfigMap"),
    "secret": ("v1", "Secret"),
    "serviceaccount": ("v1", "ServiceAccount"),
    "node": ("v1", "Node"),
    "deployment": ("apps/v1", "Deployment"),
    "replicaset": ("apps/v1", "ReplicaSet"),
    "route": ("route.openshift.io/v1", "Route"),
    "role": ("rbac.authorization.k8s.io/v1", "Role"),
    "rolebinding": ("rbac.authorization.k8s.io/v1", "RoleBinding"),
    "clusterrole": ("rbac.authorization.k8s.io/v1", "ClusterRole"),
    "clusterrolebinding": ("rbac.authorization.k8s.io/v1", "ClusterRoleBinding"),
    "olsconfig": ("ols.openshift.io/v1alpha1", "OLSConfig"),
    "clusterserviceversion": ("operators.coreos.com/v1alpha1", "ClusterServiceVersion"),
    "installplan": ("operators.coreos.com/v1alpha1", "InstallPlan"),
    "imagedigestmirrorset": ("config.openshift.io/v1", "ImageDigestMirrorSet"),
}


def resource_for(resource: str) -> Any:
    """Discover the explicit API group and kind used by a test."""
    api_version, kind = RESOURCES[resource]
    return get_client().resources.get(api_version=api_version, kind=kind)


def read_resource(
    resource: str,
    name: str | None = None,
    namespace: str | None = NAMESPACE,
    request_timeout: float = REQUEST_TIMEOUT,
    **selectors: str,
) -> dict[str, Any]:
    """Read a resource or resource list as a plain dictionary."""
    api = resource_for(resource)
    return api.get(
        name=name,
        namespace=namespace if api.namespaced else None,
        _request_timeout=request_timeout,
        **selectors,
    ).to_dict()


def create_resource(
    body: dict[str, Any], namespace: str = NAMESPACE, ignore_existing: bool = False
) -> None:
    """Create a resource, optionally tolerating only an existing resource."""
    api = get_client().resources.get(api_version=body["apiVersion"], kind=body["kind"])
    try:
        api.create(
            body=body,
            namespace=(
                body.get("metadata", {}).get("namespace", namespace)
                if api.namespaced
                else None
            ),
            _request_timeout=REQUEST_TIMEOUT,
        )
    except ConflictError:
        if not ignore_existing:
            raise


def apply_resource(body: dict[str, Any], namespace: str = NAMESPACE) -> None:
    """Apply the desired test configuration using server-side field ownership."""
    body = deepcopy(body)
    body.pop("status", None)
    for field in (
        "managedFields",
        "resourceVersion",
        "uid",
        "creationTimestamp",
        "generation",
    ):
        body["metadata"].pop(field, None)
    api = get_client().resources.get(api_version=body["apiVersion"], kind=body["kind"])
    api.server_side_apply(
        body=body,
        name=body["metadata"]["name"],
        namespace=(
            body["metadata"].get("namespace", namespace) if api.namespaced else None
        ),
        field_manager="ols-e2e",
        force_conflicts=False,
        _request_timeout=REQUEST_TIMEOUT,
    )


def apply_file(path: str, namespace: str = NAMESPACE) -> None:
    """Apply every nonempty YAML document in a manifest file."""
    for body in yaml.safe_load_all(Path(path).read_text(encoding="utf-8")):
        if body:
            apply_resource(body, namespace)


def patch_resource(
    resource: str,
    name: str,
    body: dict | list,
    namespace: str = NAMESPACE,
    patch_type: str = "merge",
) -> None:
    """Patch a resource with a structured merge or JSON patch."""
    api = resource_for(resource)
    api.patch(
        name=name,
        namespace=namespace if api.namespaced else None,
        body=body,
        content_type=f"application/{patch_type}-patch+json",
        _request_timeout=REQUEST_TIMEOUT,
    )


def delete_resource(
    resource: str,
    name: str,
    namespace: str = NAMESPACE,
    ignore_missing: bool = False,
    timeout: int = 600,
) -> None:
    """Delete a resource and wait for its finalizers to finish."""
    api = resource_for(resource)
    kwargs = {
        "name": name,
        "namespace": namespace if api.namespaced else None,
        "_request_timeout": REQUEST_TIMEOUT,
    }
    try:
        uid = api.get(**kwargs).metadata.uid
    except NotFoundError:
        if ignore_missing:
            return
        raise
    try:
        api.delete(**kwargs, body={"preconditions": {"uid": uid}})
    except NotFoundError:
        return
    except ConflictError:
        try:
            current_uid = api.get(**kwargs).metadata.uid
        except NotFoundError:
            return
        if current_uid == uid:
            raise
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if api.get(**kwargs).metadata.uid != uid:
                return
        except NotFoundError:
            return
        time.sleep(1)
    raise TimeoutError(f"Timed out deleting {resource}/{name}")


def scale_deployment(name: str, replicas: int, namespace: str = NAMESPACE) -> None:
    """Set the desired replica count of a deployment."""
    patch_resource("deployment", name, {"spec": {"replicas": replicas}}, namespace)


def create_secret(
    name: str,
    values: dict[str, str],
    namespace: str = NAMESPACE,
    ignore_existing: bool = True,
) -> None:
    """Create an opaque secret without putting credentials on a command line."""
    create_resource(
        {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": name},
            "type": "Opaque",
            "data": {
                key: base64.b64encode(value.encode()).decode()
                for key, value in values.items()
            },
        },
        namespace,
        ignore_existing,
    )


def _create_or_validate_rbac(body: dict[str, Any], namespace: str) -> None:
    """Accept an existing RBAC resource only when its permissions match."""
    try:
        create_resource(body, namespace)
    except ConflictError:
        existing = read_resource(
            body["kind"].lower(), body["metadata"]["name"], namespace
        )
        fields = ("rules",) if body["kind"] == "Role" else ("roleRef", "subjects")
        if any(existing.get(field) != body[field] for field in fields):
            raise RuntimeError(
                f"Existing {body['kind']}/{body['metadata']['name']} has unexpected permissions"
            ) from None


def configure_pod_reader(namespace: str = NAMESPACE) -> None:
    """Grant the test service account permission to read pods."""
    _create_or_validate_rbac(
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {"name": "pod-reader"},
            "rules": [
                {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list"]}
            ],
        },
        namespace,
    )
    _create_or_validate_rbac(
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {"name": "test-user-pod-reader"},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": "pod-reader",
            },
            "subjects": [
                {"kind": "ServiceAccount", "name": "test-user", "namespace": namespace}
            ],
        },
        namespace,
    )


def warm_image(node: str, image: str, timeout: int = 600) -> None:
    """Pull an image on a node using a short-lived, unprivileged pod."""
    name = f"ols-image-warmup-{uuid4().hex[:12]}"
    pod_body = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": name},
        "spec": {
            "nodeName": node,
            "restartPolicy": "Never",
            "activeDeadlineSeconds": timeout,
            "automountServiceAccountToken": False,
            "tolerations": [{"operator": "Exists"}],
            "securityContext": {
                "runAsNonRoot": True,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "containers": [
                {
                    "name": "warmup",
                    "image": image,
                    "imagePullPolicy": "Always",
                    "command": ["/bin/true"],
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                    },
                }
            ],
        },
    }
    try:
        create_resource(pod_body)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pod = read_resource("pod", name)
            statuses = pod.get("status", {}).get("containerStatuses", [])
            if any(status.get("imageID") for status in statuses):
                return
            time.sleep(2)
        raise TimeoutError(f"Timed out pulling image on node {node}")
    finally:
        delete_resource("pod", name, ignore_missing=True, timeout=60)


def execute(
    pod: str,
    command: list[str],
    namespace: str = NAMESPACE,
    container: str | None = None,
    timeout: int = REQUEST_TIMEOUT,
) -> str:
    """Execute a command with a deadline and a fixed 120-second connection timeout."""
    if container is None:
        pod_spec = read_resource("pod", pod, namespace)
        container = (
            pod_spec["metadata"]
            .get("annotations", {})
            .get(
                "kubectl.kubernetes.io/default-container",
                pod_spec["spec"]["containers"][0]["name"],
            )
        )
    deadline = time.monotonic() + timeout
    with (
        closing(client.ApiClient(get_client().client.configuration)) as transport,
        closing(
            stream(
                client.CoreV1Api(transport).connect_get_namespaced_pod_exec,
                pod,
                namespace,
                command=command,
                container=container,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _preload_content=False,
                _request_timeout=timeout,
            )
        ) as connection,
    ):
        stdout, stderr = [], []
        while connection.is_open():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out executing command in pod {pod}")
            connection.sock.settimeout(remaining)
            connection.update(timeout=min(1, remaining))
            stdout.append(connection.read_stdout(timeout=0))
            stderr.append(connection.read_stderr(timeout=0))
        stdout.append(connection.read_stdout(timeout=0))
        stderr.append(connection.read_stderr(timeout=0))
        if connection.returncode != 0:
            raise RuntimeError(f"Pod {pod} command failed: {''.join(stderr)}")
        return "".join(stdout)


def pod_logs(
    pod: str,
    container: str | None = None,
    namespace: str = NAMESPACE,
    tail_lines: int | None = None,
) -> str:
    """Read a pod's logs without following the stream."""
    return client.CoreV1Api(get_client().client).read_namespaced_pod_log(
        name=pod,
        namespace=namespace,
        container=container,
        tail_lines=tail_lines,
        follow=False,
        _request_timeout=REQUEST_TIMEOUT,
    )


def deployment_logs(name: str, namespace: str = NAMESPACE) -> str:
    """Read a non-terminating pod's logs, preferring ready pods then newer pods."""
    deployment = read_resource("deployment", name, namespace)
    selector = deployment["spec"]["selector"]
    if selector.get("matchExpressions"):
        raise RuntimeError(
            f"Deployment {name} selector uses matchExpressions, which "
            "deployment_logs does not support"
        )
    labels = selector["matchLabels"]
    pods = read_resource(
        "pod",
        namespace=namespace,
        label_selector=",".join(f"{key}={value}" for key, value in labels.items()),
        field_selector="status.phase=Running",
    )["items"]
    pods = [pod for pod in pods if not pod["metadata"].get("deletionTimestamp")]
    if not pods:
        raise RuntimeError(f"No non-terminating running pods for deployment {name}")

    def selection_key(pod: dict[str, Any]) -> tuple[bool, str, str]:
        ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in pod.get("status", {}).get("conditions", [])
        )
        return (
            ready,
            pod["metadata"].get("creationTimestamp", ""),
            pod["metadata"]["name"],
        )

    pod = max(pods, key=selection_key)
    container = (
        pod["metadata"]
        .get("annotations", {})
        .get(
            "kubectl.kubernetes.io/default-container",
            pod["spec"]["containers"][0]["name"],
        )
    )
    return pod_logs(pod["metadata"]["name"], container, namespace)


def wait_for_rollout(name: str, namespace: str = NAMESPACE, timeout: int = 90) -> None:
    """Wait until the deployment's current generation is fully available."""
    deadline = time.monotonic() + timeout
    while (remaining := deadline - time.monotonic()) > 0:
        deployment = read_resource(
            "deployment",
            name,
            namespace,
            request_timeout=min(REQUEST_TIMEOUT, remaining),
        )
        status = deployment.get("status", {})
        replicas = deployment["spec"].get("replicas", 1)
        if (
            status.get("observedGeneration", 0) >= deployment["metadata"]["generation"]
            and status.get("updatedReplicas", 0) == replicas
            and status.get("replicas", 0) == replicas
            and status.get("availableReplicas", 0) == replicas
        ):
            return
        time.sleep(max(0, min(1, deadline - time.monotonic())))
    raise TimeoutError(f"Timed out waiting for deployment {name}")
