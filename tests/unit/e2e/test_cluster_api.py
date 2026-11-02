"""Test native cluster operations, including errors, waits and remote execution."""

import base64
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from kubernetes.client import ApiException
from kubernetes.stream.ws_client import WSClient
from openshift.dynamic.exceptions import ConflictError, ForbiddenError, NotFoundError

from tests.e2e.utils import cluster_api as api


@pytest.fixture
def resource() -> Iterator[MagicMock]:
    """Discover a fake namespaced resource and reject subprocess fallback."""
    with (
        patch.object(api, "get_client") as client,
        patch("subprocess.run", side_effect=AssertionError("CLI fallback")),
    ):
        resource = client.return_value.resources.get.return_value
        resource.namespaced = True
        yield resource


@pytest.mark.parametrize("namespaced", [True, False])
def test_read_resource_scope(resource: MagicMock, namespaced: bool) -> None:
    """Send namespace and selectors only to the intended resource scope."""
    resource.namespaced = namespaced
    resource.get.return_value.to_dict.return_value = {"items": []}
    assert api.read_resource("pod", namespace="custom", label_selector="app=ols") == {
        "items": []
    }
    resource.get.assert_called_once_with(
        name=None,
        namespace="custom" if namespaced else None,
        label_selector="app=ols",
        _request_timeout=120,
    )


@pytest.mark.parametrize("ignore", [True, False])
def test_create_resource_conflict(resource: MagicMock, ignore: bool) -> None:
    """Ignore an already-existing resource only when explicitly requested."""
    resource.create.side_effect = ConflictError(ApiException(status=409))
    body = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "test", "namespace": "explicit"},
    }
    if ignore:
        api.create_resource(body, ignore_existing=True)
    else:
        with pytest.raises(ConflictError):
            api.create_resource(body)
    assert resource.create.call_args.kwargs["namespace"] == "explicit"


def test_create_resource_forbidden_propagates(resource: MagicMock) -> None:
    """Never suppress authorization failures as existing resources."""
    resource.create.side_effect = ForbiddenError(ApiException(status=403))
    with pytest.raises(ForbiddenError):
        api.create_resource(
            {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": "test"}},
            ignore_existing=True,
        )


@pytest.mark.parametrize("namespaced", [True, False])
def test_apply_resource_strips_server_fields(
    resource: MagicMock, namespaced: bool
) -> None:
    """Apply a fetched resource without invalid managed fields or mutating its caller."""
    resource.namespaced = namespaced
    body = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "config",
            "namespace": "explicit",
            "managedFields": [],
            "uid": "uid",
            "resourceVersion": "1",
        },
        "status": {},
        "data": {"key": "value"},
    }
    api.apply_resource(body)
    kwargs = resource.server_side_apply.call_args.kwargs
    assert kwargs["body"]["metadata"] == {"name": "config", "namespace": "explicit"}
    assert "status" not in kwargs["body"]
    assert "uid" in body["metadata"]
    assert kwargs["namespace"] == ("explicit" if namespaced else None)
    assert kwargs["field_manager"] == "ols-e2e"
    assert kwargs["force_conflicts"] is False
    assert kwargs["_request_timeout"] == 120


def test_apply_file_multiple_documents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Apply every document, skipping empty YAML separators."""
    path = tmp_path / "resources.yaml"
    path.write_text("kind: ConfigMap\n---\n---\nkind: Secret\n")
    apply = Mock()
    monkeypatch.setattr(api, "apply_resource", apply)
    api.apply_file(str(path), "custom")
    assert [call.args for call in apply.call_args_list] == [
        ({"kind": "ConfigMap"}, "custom"),
        ({"kind": "Secret"}, "custom"),
    ]


@pytest.mark.parametrize("patch_type", ["json", "merge"])
def test_patch_resource_content_type(resource: MagicMock, patch_type: str) -> None:
    """Use the correct Kubernetes patch media type."""
    body = (
        [{"op": "remove", "path": "/data/key"}]
        if patch_type == "json"
        else {"data": {"key": None}}
    )
    api.patch_resource("configmap", "config", body, "custom", patch_type)
    resource.patch.assert_called_once_with(
        name="config",
        namespace="custom",
        body=body,
        content_type=f"application/{patch_type}-patch+json",
        _request_timeout=120,
    )


@pytest.mark.parametrize("recreated", [True, False])
def test_delete_resource_waits_for_original_uid(
    resource: MagicMock, monkeypatch: pytest.MonkeyPatch, recreated: bool
) -> None:
    """Finish when the original resource disappears, even if a controller recreates it."""
    original = SimpleNamespace(metadata=SimpleNamespace(uid="old"))
    gone = (
        SimpleNamespace(metadata=SimpleNamespace(uid="new"))
        if recreated
        else NotFoundError(ApiException(status=404))
    )
    resource.get.side_effect = [original, original, gone]
    sleep = Mock()
    monkeypatch.setattr(api.time, "sleep", sleep)
    api.delete_resource("secret", "cert", "custom")
    resource.delete.assert_called_once_with(
        name="cert",
        namespace="custom",
        body={"preconditions": {"uid": "old"}},
        _request_timeout=120,
    )
    sleep.assert_called_once_with(1)


@pytest.mark.parametrize("ignore,status", [(True, 404), (False, 404), (True, 403)])
def test_delete_resource_only_ignores_missing(
    resource: MagicMock, ignore: bool, status: int
) -> None:
    """Propagate deletion access errors and explicitly requested missing-resource errors."""
    error = NotFoundError if status == 404 else ForbiddenError
    resource.get.side_effect = error(ApiException(status=status))
    if ignore and status == 404:
        api.delete_resource("secret", "test", ignore_missing=True)
    else:
        with pytest.raises(error):
            api.delete_resource("secret", "test", ignore_missing=ignore)
    resource.delete.assert_not_called()


def test_delete_resource_timeout(resource: MagicMock) -> None:
    """Fail if a finalizer prevents completion."""
    with pytest.raises(TimeoutError, match="deleting secret/test"):
        api.delete_resource("secret", "test", timeout=0)


def test_scale_deployment(resource: MagicMock) -> None:
    """Patch replicas rather than replacing the deployment."""
    api.scale_deployment("server", 0, "custom")
    assert resource.patch.call_args.kwargs["body"] == {"spec": {"replicas": 0}}


def test_create_secret_encodes_values(resource: MagicMock) -> None:
    """Preserve newlines and Unicode credential contents."""
    api.create_secret("creds", {"key": "test-ä\n"}, "custom")
    body = resource.create.call_args.kwargs["body"]
    assert body["type"] == "Opaque"
    assert base64.b64decode(body["data"]["key"]).decode() == "test-ä\n"


def test_configure_pod_reader(resource: MagicMock) -> None:
    """Bind the namespaced pod reader to the correct service account."""
    api.configure_pod_reader("custom")
    role, binding = [call.kwargs["body"] for call in resource.create.call_args_list]
    assert role["rules"] == [
        {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list"]}
    ]
    assert binding["subjects"] == [
        {"kind": "ServiceAccount", "name": "test-user", "namespace": "custom"}
    ]
    assert binding["roleRef"]["name"] == "pod-reader"


@pytest.mark.parametrize("outcome", ["success", "timeout", "error", "create-error"])
def test_warm_image_always_cleans_up(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """Delete the bounded unprivileged pull pod even when pulling fails."""
    create, delete = Mock(), Mock()
    read = Mock(
        side_effect=[
            {"status": {}},
            {"status": {"containerStatuses": [{"imageID": "sha256:image"}]}},
        ]
    )
    if outcome == "error":
        read.side_effect = RuntimeError("read failed")
    if outcome == "create-error":
        create.side_effect = RuntimeError("create failed")
    monkeypatch.setattr(api, "create_resource", create)
    monkeypatch.setattr(api, "delete_resource", delete)
    monkeypatch.setattr(api, "read_resource", read)
    monkeypatch.setattr(api.time, "sleep", Mock())
    if outcome == "success":
        api.warm_image("worker", "image")
        assert read.call_count == 2
    else:
        with pytest.raises(TimeoutError if outcome == "timeout" else RuntimeError):
            api.warm_image(
                "worker", "image", timeout=0 if outcome == "timeout" else 600
            )
    body = create.call_args.args[0]
    assert body["spec"]["nodeName"] == "worker"
    assert body["spec"]["containers"][0]["image"] == "image"
    assert body["spec"]["automountServiceAccountToken"] is False
    delete.assert_called_once_with(
        "pod", body["metadata"]["name"], ignore_missing=True, timeout=60
    )


@pytest.mark.parametrize("outcome", ["success", "exit-error", "timeout"])
def test_execute_checks_remote_status(
    resource: MagicMock, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """Bound execution and propagate nonzero remote exits without a subprocess."""
    connection = MagicMock(spec=WSClient)
    connection.sock = Mock()
    connection.is_open.side_effect = [True, False]
    connection.read_stdout.side_effect = ["output", "tail"]
    connection.read_stderr.return_value = "remote error"
    connection.returncode = 1 if outcome == "exit-error" else 0
    websocket = Mock(return_value=connection)
    resource.get.return_value.to_dict.return_value = {
        "metadata": {
            "annotations": {"kubectl.kubernetes.io/default-container": "server"}
        },
        "spec": {"containers": [{"name": "sidecar"}, {"name": "server"}]},
    }
    monkeypatch.setattr(api, "stream", websocket)
    monkeypatch.setattr(api.client, "ApiClient", Mock())
    monkeypatch.setattr(api.client, "CoreV1Api", Mock())
    if outcome == "success":
        assert (
            api.execute("pod", ["ls", "/data"], "custom", timeout=600) == "outputtail"
        )
        assert websocket.call_args.kwargs["_request_timeout"] == 600
    else:
        with pytest.raises(TimeoutError if outcome == "timeout" else RuntimeError):
            api.execute("pod", ["false"], timeout=0 if outcome == "timeout" else 120)
    assert websocket.call_args.kwargs["container"] == "server"
    assert all(
        call.kwargs["timeout"] == 0
        for call in connection.read_stdout.call_args_list
        + connection.read_stderr.call_args_list
    )
    connection.close.assert_called_once()


def test_execute_timeout_survives_silent_command(
    resource: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enforce the deadline for a command that runs without writing any output."""

    class SilentConnection:
        """Model WSClient channel reads, which block when given no timeout."""

        returncode = 0
        sock = Mock()

        def is_open(self) -> bool:
            """Keep the silent command running until its deadline."""
            return True

        def update(self, timeout: float | None = 0) -> None:
            """Require bounded polling without producing channel data."""
            assert timeout is not None

        def read_stdout(self, timeout: float | None = None) -> str:
            """Reject channel reads that would wait indefinitely."""
            if timeout is None:
                pytest.fail("Channel read would block past the execution deadline")
            return ""

        read_stderr = read_stdout

        def close(self) -> None:
            """Ignore the close of a connection that was never opened."""

    resource.get.return_value.to_dict.return_value = {
        "metadata": {},
        "spec": {"containers": [{"name": "server"}]},
    }
    monkeypatch.setattr(api, "stream", Mock(return_value=SilentConnection()))
    monkeypatch.setattr(api.client, "ApiClient", Mock())
    monkeypatch.setattr(api.client, "CoreV1Api", Mock())
    with pytest.raises(TimeoutError, match="Timed out executing command in pod"):
        api.execute("pod", ["sleep", "600"], timeout=1)


def test_execute_deadline_includes_handshake(
    resource: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not start a fresh execution timeout after a slow handshake."""
    connection = MagicMock(spec=WSClient)
    connection.is_open.return_value = True
    monkeypatch.setattr(api, "stream", Mock(return_value=connection))
    monkeypatch.setattr(api.client, "ApiClient", Mock())
    monkeypatch.setattr(api.client, "CoreV1Api", Mock())
    monkeypatch.setattr(api.time, "monotonic", Mock(side_effect=[0, 2]))
    with pytest.raises(TimeoutError, match="Timed out executing command"):
        api.execute("pod", ["true"], container="server", timeout=1)
    connection.update.assert_not_called()
    connection.close.assert_called_once()


def test_pod_logs(resource: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward container and tail selection to the log endpoint."""
    core = Mock()
    monkeypatch.setattr(api.client, "CoreV1Api", core)
    core.return_value.read_namespaced_pod_log.return_value = "logs\n"
    assert api.pod_logs("pod", "server", "custom", tail_lines=50) == "logs\n"
    core.return_value.read_namespaced_pod_log.assert_called_once_with(
        name="pod",
        namespace="custom",
        container="server",
        tail_lines=50,
        follow=False,
        _request_timeout=120,
    )


@pytest.mark.parametrize("has_pods", [True, False])
def test_deployment_logs_uses_selector(
    monkeypatch: pytest.MonkeyPatch, has_pods: bool
) -> None:
    """Select a running pod and honor its default-container annotation."""
    pod = {
        "metadata": {
            "name": "pod",
            "annotations": {"kubectl.kubernetes.io/default-container": "server"},
        },
        "spec": {"containers": [{"name": "sidecar"}, {"name": "server"}]},
    }
    read = Mock(
        side_effect=[
            {"spec": {"selector": {"matchLabels": {"app": "server"}}}},
            {"items": [pod] if has_pods else []},
        ]
    )
    logs = Mock(return_value="logs")
    monkeypatch.setattr(api, "read_resource", read)
    monkeypatch.setattr(api, "pod_logs", logs)
    if has_pods:
        assert api.deployment_logs("server", "custom") == "logs"
        logs.assert_called_once_with("pod", "server", "custom")
    else:
        with pytest.raises(RuntimeError, match="No non-terminating running pods"):
            api.deployment_logs("server", "custom")
    assert read.call_args.kwargs["label_selector"] == "app=server"
    assert read.call_args.kwargs["field_selector"] == "status.phase=Running"


def test_deployment_logs_rejects_match_expressions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail loudly rather than pick the wrong pod for expression selectors."""
    read = Mock(
        return_value={
            "spec": {
                "selector": {
                    "matchExpressions": [
                        {"key": "app", "operator": "In", "values": ["server"]}
                    ]
                }
            }
        }
    )
    monkeypatch.setattr(api, "read_resource", read)
    with pytest.raises(RuntimeError, match="matchExpressions"):
        api.deployment_logs("server")


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "candidates,expected",
    [
        (
            [("old", True, "1", True), ("new", True, "2", False)],
            "new",
        ),
        (
            [("old", True, "1", False), ("new", False, "2", False)],
            "old",
        ),
        (
            [("old", True, "1", False), ("new", True, "2", False)],
            "new",
        ),
        (
            [("old", False, "1", False), ("new", False, "2", False)],
            "new",
        ),
        (
            [("pod-a", True, "1", False), ("pod-b", True, "1", False)],
            "pod-b",
        ),
    ],
    ids=["terminating", "readiness", "newest-ready", "unready-fallback", "tie"],
)
def test_deployment_logs_selection_is_independent_of_list_order(
    monkeypatch: pytest.MonkeyPatch,
    reverse: bool,
    candidates: list[tuple[str, bool, str, bool]],
    expected: str,
) -> None:
    """Prefer active ready pods during rollouts regardless of API list ordering."""
    pods = [
        {
            "metadata": {
                "name": name,
                "creationTimestamp": f"2026-01-01T00:00:0{created}Z",
                **(
                    {"deletionTimestamp": "2026-01-01T00:01:00Z"} if terminating else {}
                ),
            },
            "spec": {"containers": [{"name": "server"}]},
            "status": {
                "conditions": [
                    {"type": "Ready", "status": "True" if ready else "False"}
                ]
            },
        }
        for name, ready, created, terminating in candidates
    ]
    monkeypatch.setattr(
        api,
        "read_resource",
        Mock(
            side_effect=[
                {"spec": {"selector": {"matchLabels": {"app": "server"}}}},
                {"items": list(reversed(pods)) if reverse else pods},
            ]
        ),
    )
    logs = Mock(return_value="logs")
    monkeypatch.setattr(api, "pod_logs", logs)

    assert api.deployment_logs("server", "custom") == "logs"
    logs.assert_called_once_with(expected, "server", "custom")


def test_deployment_logs_rejects_only_terminating_pods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never use a terminating pod as evidence for requests to its replacement."""
    monkeypatch.setattr(
        api,
        "read_resource",
        Mock(
            side_effect=[
                {"spec": {"selector": {"matchLabels": {"app": "server"}}}},
                {
                    "items": [
                        {"metadata": {"deletionTimestamp": "2026-01-01T00:00:00Z"}}
                    ]
                },
            ]
        ),
    )
    logs = Mock()
    monkeypatch.setattr(api, "pod_logs", logs)

    with pytest.raises(RuntimeError, match="No non-terminating running pods"):
        api.deployment_logs("server")
    logs.assert_not_called()


@pytest.mark.parametrize("timeout", [0, 90])
def test_wait_for_rollout_checks_generation(
    monkeypatch: pytest.MonkeyPatch, timeout: int
) -> None:
    """Wait beyond stale status and check all desired replicas are updated."""
    deployment = {
        "metadata": {"generation": 2},
        "spec": {"replicas": 1},
        "status": {
            "observedGeneration": 2,
            "updatedReplicas": 1,
            "replicas": 1,
            "availableReplicas": 1,
        },
    }
    read = Mock(side_effect=[{**deployment, "status": {}}, deployment])
    sleep = Mock()
    monkeypatch.setattr(api, "read_resource", read)
    monkeypatch.setattr(api.time, "sleep", sleep)
    if timeout:
        api.wait_for_rollout("server", timeout=timeout)
        sleep.assert_called_once_with(1)
    else:
        with pytest.raises(TimeoutError):
            api.wait_for_rollout("server", timeout=timeout)


def test_resource_discovery_uses_explicit_api_version() -> None:
    """Discover the intended API group rather than an ambiguous short name."""
    with patch.object(api, "get_client") as client:
        api.resource_for("deployment")
        client.return_value.resources.get.assert_called_once_with(
            api_version="apps/v1", kind="Deployment"
        )
