"""Cover cluster lifecycle races and bounded waits without a live cluster."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from kubernetes.client.exceptions import ApiException
from openshift.dynamic.exceptions import ConflictError, ForbiddenError, NotFoundError

from tests.e2e import conftest as e2e_fixtures
from tests.e2e.utils import adapt_ols_config, cluster
from tests.e2e.utils import cluster_api as api


@pytest.mark.parametrize("ignore_missing", [False, True])
@pytest.mark.parametrize("replacement", ["new", "old", None])
def test_delete_conflict_checks_original_uid(
    monkeypatch: pytest.MonkeyPatch, replacement: str | None, ignore_missing: bool
) -> None:
    """Suppress conflicts only when the original object is confirmed gone."""
    resource = Mock(namespaced=True)
    error = ConflictError(ApiException(status=409))
    resource.delete.side_effect = error
    original = SimpleNamespace(metadata=SimpleNamespace(uid="old"))
    after = (
        SimpleNamespace(metadata=SimpleNamespace(uid=replacement))
        if replacement
        else NotFoundError(ApiException(status=404))
    )
    resource.get.side_effect = [original, after]
    monkeypatch.setattr(api, "resource_for", Mock(return_value=resource))
    if replacement == "old":
        with pytest.raises(ConflictError) as raised:
            api.delete_resource("pod", "test", ignore_missing=ignore_missing)
        assert raised.value is error
    else:
        api.delete_resource("pod", "test", ignore_missing=ignore_missing)
    assert resource.get.call_count == 2


@pytest.mark.parametrize("ignore_missing", [False, True])
def test_delete_accepts_missing_after_initial_read(
    monkeypatch: pytest.MonkeyPatch, ignore_missing: bool
) -> None:
    """Treat a DELETE 404 as success once the original object was observed."""
    resource = Mock(namespaced=True)
    resource.get.return_value = SimpleNamespace(metadata=SimpleNamespace(uid="old"))
    resource.delete.side_effect = NotFoundError(ApiException(status=404))
    monkeypatch.setattr(api, "resource_for", Mock(return_value=resource))
    api.delete_resource("pod", "test", ignore_missing=ignore_missing)
    resource.get.assert_called_once()
    resource.delete.assert_called_once()


def test_delete_conflict_does_not_hide_read_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve errors from the initial read instead of treating them as deletion races."""
    resource = Mock(namespaced=True)
    resource.get.side_effect = ConflictError(ApiException(status=409))
    monkeypatch.setattr(api, "resource_for", Mock(return_value=resource))
    with pytest.raises(ConflictError):
        api.delete_resource("pod", "test", ignore_missing=True)
    resource.delete.assert_not_called()


@pytest.mark.parametrize("mismatch", [None, "rules", "roleRef", "subjects"])
def test_existing_pod_reader_permissions_are_validated(
    monkeypatch: pytest.MonkeyPatch, mismatch: str | None
) -> None:
    """Accept matching RBAC and reject stale rules, bindings, or subjects."""
    bodies = {}

    def create(body: dict, _namespace: str) -> None:
        bodies[body["kind"].lower()] = body
        raise ConflictError(ApiException(status=409))

    def read(kind: str, _name: str, _namespace: str) -> dict:
        assert kind in api.RESOURCES
        body = dict(bodies[kind])
        if mismatch in body:
            body[mismatch] = []
        return body

    monkeypatch.setattr(api, "create_resource", create)
    monkeypatch.setattr(api, "read_resource", read)
    if mismatch:
        with pytest.raises(RuntimeError, match="unexpected permissions"):
            api.configure_pod_reader()
    else:
        api.configure_pod_reader()


def test_rollout_limits_request_to_remaining_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep polling unavailable replicas without exceeding the remaining request budget."""
    read = Mock(
        return_value={
            "metadata": {"generation": 2},
            "spec": {"replicas": 1},
            "status": {
                "observedGeneration": 2,
                "updatedReplicas": 1,
                "replicas": 1,
                "availableReplicas": 0,
            },
        }
    )
    monkeypatch.setattr(api, "read_resource", read)
    monkeypatch.setattr(api.time, "monotonic", Mock(side_effect=[0, 1, 2, 3, 4, 5]))
    monkeypatch.setattr(api.time, "sleep", Mock())
    with pytest.raises(TimeoutError):
        api.wait_for_rollout("server", timeout=5)
    assert read.call_args_list == [
        call("deployment", "server", api.NAMESPACE, request_timeout=4),
        call("deployment", "server", api.NAMESPACE, request_timeout=2),
    ]


def test_read_resource_honors_custom_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward the caller's timeout to the HTTP request."""
    resource = Mock(namespaced=True)
    monkeypatch.setattr(api, "resource_for", Mock(return_value=resource))
    api.read_resource("pod", "server", request_timeout=0.5)
    assert resource.get.call_args.kwargs["_request_timeout"] == 0.5


@pytest.mark.parametrize("setup_fails", [False, True])
def test_session_token_cleanup_tracks_partial_setup(
    monkeypatch: pytest.MonkeyPatch, setup_fails: bool
) -> None:
    """Remove token Secrets even when creation reports failure after reaching the server."""
    create = Mock(side_effect=RuntimeError("create failed") if setup_fails else None)
    delete = Mock()
    monkeypatch.setattr(api, "create_resource", create)
    monkeypatch.setattr(
        api, "read_resource", Mock(return_value={"data": {"token": "dGVzdA=="}})
    )
    monkeypatch.setattr(api, "delete_resource", delete)
    if setup_fails:
        with pytest.raises(RuntimeError):
            cluster.get_token_for("test-user")
    else:
        assert cluster.get_token_for("test-user") == "test"
    name = create.call_args.args[0]["metadata"]["name"]
    cluster.cleanup_session_tokens()
    delete.assert_called_once_with("secret", name, ignore_missing=True, timeout=60)
    cluster.cleanup_session_tokens()
    assert delete.call_count == 1


def test_token_cleanup_attempts_all_secrets_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry failed cleanup later without retaining successfully deleted tokens."""
    monkeypatch.setattr(cluster, "_SESSION_TOKEN_SECRETS", {"first", "second"})
    delete = Mock(side_effect=[ForbiddenError(ApiException(status=403)), None])
    monkeypatch.setattr(api, "delete_resource", delete)
    with pytest.raises(ExceptionGroup):
        cluster.cleanup_session_tokens()
    assert delete.call_count == 2
    assert len(cluster._SESSION_TOKEN_SECRETS) == 1
    delete.side_effect = None
    cluster.cleanup_session_tokens()
    assert not cluster._SESSION_TOKEN_SECRETS


@pytest.mark.parametrize("failure", ["token", "gather", "client"])
def test_session_start_failure_cleans_created_tokens(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Clean actual tracked tokens when startup, diagnostics, or client setup fails."""
    monkeypatch.setenv("OLS_URL", "")
    monkeypatch.setenv("SKIP_CLUSTER_SETUP", "true")
    monkeypatch.delenv("SWITCH_DEFAULT_PROVIDER", raising=False)
    monkeypatch.setattr(e2e_fixtures, "on_cluster", False)
    monkeypatch.setattr(e2e_fixtures, "_maybe_setup_mcp", Mock())
    monkeypatch.setattr(e2e_fixtures.ols_installer, "warm_up_rhokp_image", Mock())
    monkeypatch.setattr(
        cluster, "get_ols_url", Mock(return_value="https://ols.example")
    )
    monkeypatch.setattr(api, "create_resource", Mock())
    token = {"data": {"token": "dGVzdA=="}}
    read = Mock(
        side_effect=[
            token,
            (
                token
                if failure == "client"
                else ForbiddenError(ApiException(status=403, reason="token"))
            ),
        ]
    )
    monkeypatch.setattr(api, "read_resource", read)
    delete = Mock()
    monkeypatch.setattr(api, "delete_resource", delete)
    monkeypatch.setattr(
        e2e_fixtures,
        "must_gather",
        Mock(side_effect=RuntimeError("gather") if failure == "gather" else None),
    )
    monkeypatch.setattr(
        e2e_fixtures.client_utils,
        "get_http_client",
        Mock(side_effect=RuntimeError("client")),
    )
    with pytest.raises((RuntimeError, ForbiddenError), match=failure):
        e2e_fixtures.pytest_sessionstart()
    assert delete.call_count == 2
    assert not cluster._SESSION_TOKEN_SECRETS
    assert all(
        c.args[0] == "secret" and c.kwargs["ignore_missing"]
        for c in delete.call_args_list
    )


@pytest.mark.parametrize("failure", [None, "gather", "mcp"])
def test_session_finish_cleans_tokens_when_namespace_is_retained(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    """Always clean up session credentials even if artifact or MCP cleanup fails."""
    monkeypatch.setenv("SKIP_CLUSTER_SETUP", "true")
    monkeypatch.setenv("OLS_CONFIG_SUFFIX", "mcp")
    monkeypatch.setattr(e2e_fixtures, "on_cluster", True)
    monkeypatch.setattr(
        e2e_fixtures, "_should_gather_artifacts", Mock(return_value=True)
    )
    for name, stage in [("must_gather", "gather"), ("teardown_mcp_on_cluster", "mcp")]:
        monkeypatch.setattr(
            e2e_fixtures,
            name,
            Mock(side_effect=RuntimeError(stage) if failure == stage else None),
        )
    cleanup = Mock()
    monkeypatch.setattr(cluster, "cleanup_session_tokens", cleanup)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            e2e_fixtures.pytest_sessionfinish(Mock(), 0)
    else:
        e2e_fixtures.pytest_sessionfinish(Mock(), 0)
    cleanup.assert_called_once_with()


@pytest.mark.parametrize("ready", [False, True])
def test_reconcile_checks_operator_readiness_before_deleting(
    monkeypatch: pytest.MonkeyPatch, ready: bool
) -> None:
    """Do not delete the OLSConfig until the operator can process its finalizer."""
    monkeypatch.setattr(api, "scale_deployment", Mock())
    monkeypatch.setattr(cluster, "get_pod_by_prefix", Mock(return_value=["operator"]))
    monkeypatch.setattr(
        cluster,
        "get_container_ready_status",
        Mock(return_value=["true" if ready else "false"]),
    )

    def retry(_attempts: int, _interval: int, predicate, _message: str) -> bool:
        return bool(predicate())

    monkeypatch.setattr(adapt_ols_config, "retry_until_timeout_or_success", retry)
    delete = Mock(side_effect=RuntimeError("deletion reached"))
    monkeypatch.setattr(api, "delete_resource", delete)
    with pytest.raises(
        RuntimeError, match="deletion reached" if ready else "Timed out waiting"
    ):
        adapt_ols_config._reconcile_olsconfig_with_operator(["openai"])
    assert delete.call_count == int(ready)
