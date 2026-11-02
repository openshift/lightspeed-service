"""Test OLS cluster helpers backed by native API access."""

import base64
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock, Mock, patch

import pytest
from kubernetes.client import ApiException
from kubernetes.dynamic.resource import ResourceInstance
from openshift.dynamic.exceptions import ConflictError, ForbiddenError, NotFoundError

from tests.e2e.utils import cluster, cluster_api


@pytest.fixture
def resource() -> Iterator[MagicMock]:
    """Replace API discovery and prohibit CLI fallback."""
    with (
        patch.object(cluster_api, "resource_for") as discover,
        patch("subprocess.run", side_effect=AssertionError("Unexpected CLI call")),
    ):
        yield discover.return_value


def test_list_path_missing_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat an uncreated data-export directory as empty."""
    execute = Mock(return_value="")
    monkeypatch.setattr(cluster_api, "execute", execute)
    path = "/app-root/ols-user-data/feedback"
    assert cluster.list_path("service-pod", path) == []
    execute.assert_called_once_with(
        "service-pod",
        ["sh", "-c", 'if [ -e "$1" ] || [ -L "$1" ]; then ls "$1"; fi', "--", path],
    )


def test_grant_access_binding_is_owned_by_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Garbage-collect cluster-scoped test bindings when the namespace is removed."""
    monkeypatch.setattr(
        cluster_api,
        "read_resource",
        Mock(return_value={"metadata": {"uid": "namespace-uid"}}),
    )
    apply = Mock()
    monkeypatch.setattr(cluster_api, "apply_resource", apply)
    cluster.grant_sa_user_access("test-user", "query-access")
    body = apply.call_args.args[0]
    assert body["roleRef"]["name"] == "query-access"
    assert body["subjects"] == [
        {
            "kind": "ServiceAccount",
            "name": "test-user",
            "namespace": "openshift-lightspeed",
        }
    ]
    assert body["metadata"]["ownerReferences"] == [
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "name": "openshift-lightspeed",
            "uid": "namespace-uid",
        }
    ]


def test_create_file_passes_content_as_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep quotes and shell metacharacters out of the shell program."""
    execute = Mock()
    monkeypatch.setattr(cluster_api, "execute", execute)
    content = "'$(touch /tmp/unwanted)'"
    cluster.create_file("pod", "/data/file", content)
    assert execute.call_args.args == (
        "pod",
        ["sh", "-c", 'printf "%s\\n" "$1" > "$2"', "--", content, "/data/file"],
    )


@pytest.mark.parametrize("exists", [True, False])
def test_deployment_exists_handles_not_found(resource: MagicMock, exists: bool) -> None:
    """Treat only a missing Deployment as absent."""
    if not exists:
        resource.get.side_effect = NotFoundError(ApiException(status=404))
    assert cluster.deployment_exists("server", "custom") is exists
    resource.get.assert_called_once_with(
        name="server", namespace="custom", _request_timeout=120
    )


@pytest.mark.parametrize(
    "operation",
    [cluster.deployment_exists, cluster.get_pod_containers, cluster.create_user],
)
def test_api_forbidden_is_not_treated_as_missing(
    resource: MagicMock, operation: Callable
) -> None:
    """Propagate authorization errors without attempting to create resources."""
    resource.get.side_effect = ForbiddenError(ApiException(status=403))
    with pytest.raises(ForbiddenError):
        operation("server")
    resource.create.assert_not_called()


def test_get_cluster_details(resource: MagicMock) -> None:
    """Read cluster identity and version from structured fields."""
    with patch.object(cluster_api, "read_resource") as read:
        read.return_value = {
            "spec": {"clusterID": "cluster-id"},
            "status": {"desired": {"version": "4.20.1"}},
        }
        assert cluster.get_cluster_id() == "cluster-id"
        assert cluster.get_cluster_version() == ("4", "20")
    assert all(
        call.args == ("clusterversion", "version") for call in read.call_args_list
    )


def test_get_running_pods_filters_on_server(resource: MagicMock) -> None:
    """Return pod names with a server-side phase selector and explicit namespace."""
    resource.get.return_value = ResourceInstance(
        resource,
        {
            "kind": "PodList",
            "apiVersion": "v1",
            "items": [{"metadata": {"name": "pod-1"}}, {"metadata": {"name": "pod-2"}}],
        },
    )
    assert cluster.get_running_pods("custom") == ["pod-1", "pod-2"]
    resource.get.assert_called_once_with(
        namespace="custom", field_selector="status.phase=Running", _request_timeout=120
    )


def test_get_pod_containers_missing_pod(resource: MagicMock) -> None:
    """Preserve the diagnostics helper's missing-pod exception contract."""
    resource.get.side_effect = NotFoundError(ApiException(status=404))
    with pytest.raises(cluster.PodNotFoundError, match="Pod gone not found"):
        cluster.get_pod_containers("gone")


@pytest.mark.parametrize(
    "statuses, expected",
    [(None, []), ([{"ready": True}, {"ready": False}], ["true", "false"])],
)
def test_container_readiness_preserves_string_contract(
    resource: MagicMock, statuses: list[dict] | None, expected: list[str]
) -> None:
    """Handle pods whose status has not yet been populated."""
    resource.get.return_value = ResourceInstance(
        resource,
        {"kind": "Pod", "apiVersion": "v1", "status": {"containerStatuses": statuses}},
    )
    assert cluster.get_container_ready_status("pod", "custom") == expected


def test_create_user_existing_account(resource: MagicMock) -> None:
    """Leave existing service accounts unchanged."""
    cluster.create_user("test-user")
    resource.create.assert_not_called()


@pytest.mark.parametrize("ignore_conflict", [True, False])
def test_create_user_handles_creation_race(
    resource: MagicMock, ignore_conflict: bool
) -> None:
    """Ignore only the explicitly permitted concurrent creation conflict."""
    resource.get.side_effect = NotFoundError(ApiException(status=404))
    resource.create.side_effect = ConflictError(ApiException(status=409))
    if ignore_conflict:
        cluster.create_user("test-user", ignore_existing_resource=True)
    else:
        with pytest.raises(ConflictError):
            cluster.create_user("test-user")
    assert resource.create.call_args.kwargs["body"]["metadata"] == {"name": "test-user"}


def test_restart_deployment_patches_replicas(resource: MagicMock) -> None:
    """Scale down then up through the shared deployment helper."""
    with patch.object(cluster_api, "scale_deployment") as scale:
        cluster.restart_deployment("server", "custom")
    assert [call.args for call in scale.call_args_list] == [
        ("server", 0, "custom"),
        ("server", 1, "custom"),
    ]


def test_get_route_url(resource: MagicMock) -> None:
    """Build the HTTPS endpoint from the OpenShift Route host."""
    resource.get.return_value = ResourceInstance(
        resource, {"kind": "Route", "spec": {"host": "ols.example.com"}}
    )
    assert cluster.get_ols_url("ols") == "https://ols.example.com"
    resource.get.assert_called_once_with(
        name="ols", namespace="openshift-lightspeed", _request_timeout=120
    )


def test_get_certificate_secret_name(resource: MagicMock) -> None:
    """Read the serving certificate annotation without parsing CLI output."""
    resource.get.return_value = ResourceInstance(
        resource,
        {
            "kind": "Service",
            "metadata": {
                "annotations": {
                    "service.beta.openshift.io/serving-cert-secret-name": "serving-cert"
                }
            },
        },
    )
    assert cluster.get_certificate_secret_name("server", "custom") == "serving-cert"


def test_get_pod_containers(resource: MagicMock) -> None:
    """Return all regular container names from the Pod spec."""
    resource.get.return_value = ResourceInstance(
        resource,
        {
            "kind": "Pod",
            "spec": {"containers": [{"name": "server"}, {"name": "sidecar"}]},
        },
    )
    assert cluster.get_pod_containers("pod", "custom") == ["server", "sidecar"]


def test_delete_user(resource: MagicMock) -> None:
    """Delete the service account in the test namespace."""
    cluster.delete_user("test-user")
    resource.delete.assert_called_once_with(
        name="test-user", namespace="openshift-lightspeed", _request_timeout=120
    )


def test_create_user_missing_account(resource: MagicMock) -> None:
    """Create a missing service account with an explicit namespace and kind."""
    resource.get.side_effect = NotFoundError(ApiException(status=404))
    cluster.create_user("test-user")
    resource.create.assert_called_once_with(
        namespace="openshift-lightspeed",
        body={
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {"name": "test-user"},
        },
        _request_timeout=120,
    )


def test_get_container_log() -> None:
    """Read container logs without opening a follow stream."""
    with patch.object(cluster_api, "pod_logs", return_value="log entry\n") as logs:
        assert cluster.get_container_log("pod", "server") == "log entry\n"
        logs.assert_called_once_with("pod", "server")


def test_operator_deployment_exists(resource: MagicMock) -> None:
    """Look for the controller Deployment in the supplied namespace."""
    assert cluster.lightspeed_operator_manager_deployed("custom")
    resource.get.assert_called_once_with(
        name="lightspeed-operator-controller-manager",
        namespace="custom",
        _request_timeout=120,
    )


def test_get_token_preserves_session_lifetime() -> None:
    """Wait for a secret-backed token usable by long-running evaluation subprocesses."""
    with (
        patch.object(cluster_api, "create_resource") as create,
        patch.object(cluster_api, "read_resource") as read,
    ):
        read.return_value = {
            "data": {"token": base64.b64encode(b"test-value").decode()}
        }
        assert cluster.get_token_for("test-user") == "test-value"
        body = create.call_args.args[0]
        assert body["type"] == "kubernetes.io/service-account-token"
        assert body["metadata"]["annotations"] == {
            "kubernetes.io/service-account.name": "test-user"
        }
        read.assert_called_once_with("secret", body["metadata"]["name"])


def test_get_token_propagates_forbidden_without_retrying() -> None:
    """Preserve authorization failures instead of reporting a token timeout."""
    error = ForbiddenError(ApiException(status=403))
    with (
        patch.object(cluster_api, "create_resource"),
        patch.object(cluster_api, "read_resource", side_effect=error) as read,
        patch("tests.e2e.utils.retry.time.sleep") as sleep,
        pytest.raises(ForbiddenError) as raised,
    ):
        cluster.get_token_for("test-user")
    assert raised.value is error
    read.assert_called_once()
    sleep.assert_not_called()


def test_get_token_times_out_without_populated_secret() -> None:
    """Report a missing token rather than returning an empty credential."""
    with (
        patch.object(cluster_api, "create_resource"),
        patch.object(cluster, "retry_until_timeout_or_success", return_value=False),
        pytest.raises(TimeoutError, match="test-user"),
    ):
        cluster.get_token_for("test-user")
