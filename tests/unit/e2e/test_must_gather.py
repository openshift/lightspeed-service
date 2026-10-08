"""Unit tests for cluster artifact collection when Pods disappear."""

import subprocess
from pathlib import Path

import pytest

from tests.scripts import must_gather as gather_module


def test_must_gather_skips_disappeared_pod_but_collects_other_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Continue log collection after a Pod vanishes between listing and logs."""
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setenv("SUITE_ID", "inspection")
    monkeypatch.setattr(
        gather_module.cluster_utils,
        "run_oc_and_store_stdout",
        lambda args, path: None,
    )
    monkeypatch.setattr(
        gather_module.cluster_utils,
        "get_running_pods",
        lambda: ["vanished", "healthy"],
    )
    monkeypatch.setattr(
        gather_module.cluster_utils,
        "get_pod_containers",
        lambda pod: ["server"],
    )

    def fake_run_oc(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[1] == "pod/vanished":
            raise subprocess.CalledProcessError(
                1,
                ["oc", *args],
                stderr='Error from server (NotFound): pods "vanished" not found',
            )
        return subprocess.CompletedProcess(["oc", *args], 0, stdout="safe-log-entry")

    monkeypatch.setattr(gather_module.cluster_utils, "run_oc", fake_run_oc)

    gather_module.must_gather()

    log_dir = tmp_path / "inspection" / "cluster" / "podlogs"
    assert not (log_dir / "vanished-server.log").exists()
    assert (log_dir / "healthy-server.log").read_text() == "safe-log-entry"


def test_must_gather_does_not_ignore_unrelated_log_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Do not mask authentication or other oc failures as Pod deletion."""
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(
        gather_module.cluster_utils,
        "run_oc_and_store_stdout",
        lambda args, path: None,
    )
    monkeypatch.setattr(
        gather_module.cluster_utils, "get_running_pods", lambda: ["healthy"]
    )
    monkeypatch.setattr(
        gather_module.cluster_utils, "get_pod_containers", lambda pod: ["server"]
    )

    def fail_run_oc(args: list[str]) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            1, ["oc", *args], stderr="Forbidden: insufficient access"
        )

    monkeypatch.setattr(gather_module.cluster_utils, "run_oc", fail_run_oc)

    with pytest.raises(subprocess.CalledProcessError):
        gather_module.must_gather()
