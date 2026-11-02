"""Test artifact collection when pods disappear or log access fails."""

from pathlib import Path
from unittest.mock import Mock, call

import pytest
from kubernetes.client import ApiException
from openshift.dynamic.exceptions import (
    ForbiddenError,
    NotFoundError,
    ResourceNotFoundError,
)

from tests.scripts import must_gather as gather_module


@pytest.mark.parametrize("status", [404, 403])
def test_must_gather_handles_log_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int
) -> None:
    """Skip only disappeared pods while preserving authorization failures."""
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setenv("SUITE_ID", "inspection")
    monkeypatch.setattr(
        gather_module.cluster_api, "read_resource", Mock(return_value={"items": []})
    )
    monkeypatch.setattr(
        gather_module.cluster_utils, "get_running_pods", lambda: ["vanished", "healthy"]
    )
    monkeypatch.setattr(
        gather_module.cluster_utils,
        "get_pod_containers",
        lambda pod: ["server", "sidecar"] if pod == "vanished" else ["server"],
    )

    def fake_pod_logs(pod: str, _container: str) -> str:
        if pod == "vanished":
            raise ApiException(status=status)
        return "safe-log-entry"

    pod_logs = Mock(side_effect=fake_pod_logs)
    monkeypatch.setattr(gather_module.cluster_api, "pod_logs", pod_logs)
    if status == 403:
        with pytest.raises(ApiException) as error:
            gather_module.must_gather()
        assert error.value.status == 403
    else:
        gather_module.must_gather()
        logs = tmp_path / "inspection" / "cluster" / "podlogs"
        assert not (logs / "vanished-server.log").exists()
        assert pod_logs.call_args_list == [
            call("vanished", "server"),
            call("healthy", "server"),
        ]
        assert (logs / "healthy-server.log").read_text() == "safe-log-entry"
        assert (logs.parent / "pods.yaml").read_text() == "items: []\n"


@pytest.mark.parametrize("discovery_error", [False, True])
def test_must_gather_skips_absent_resource_types(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, discovery_error: bool
) -> None:
    """Skip resource types whose CRD is absent without stopping collection."""
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setenv("SUITE_ID", "inspection")

    def fake_read_resource(resource: str) -> dict:
        if resource == "installplan":
            if discovery_error:
                raise ResourceNotFoundError("No matches for InstallPlan")
            raise NotFoundError(ApiException(status=404))
        return {"items": []}

    monkeypatch.setattr(gather_module.cluster_api, "read_resource", fake_read_resource)
    monkeypatch.setattr(
        gather_module.cluster_utils, "get_running_pods", lambda: ["healthy"]
    )
    monkeypatch.setattr(
        gather_module.cluster_utils, "get_pod_containers", lambda _pod: ["server"]
    )
    monkeypatch.setattr(
        gather_module.cluster_api, "pod_logs", Mock(return_value="safe logs")
    )

    gather_module.must_gather()

    cluster_dir = tmp_path / "inspection" / "cluster"
    assert not (cluster_dir / "installplan.yaml").exists()
    assert (cluster_dir / "pods.yaml").read_text() == "items: []\n"
    assert (cluster_dir / "configmap.yaml").exists()
    assert (cluster_dir / "podlogs" / "healthy-server.log").read_text() == "safe logs"


def test_must_gather_propagates_resource_authorization_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Never mistake forbidden resource access for an absent API kind."""
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(
        gather_module.cluster_api,
        "read_resource",
        Mock(side_effect=ForbiddenError(ApiException(status=403))),
    )
    with pytest.raises(ForbiddenError):
        gather_module.must_gather()
