"""Unit tests for e2e cluster utilities."""

import subprocess

import pytest

from tests.e2e.utils import cluster


def test_list_path_returns_empty_list_when_directory_does_not_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat an uncreated data-export directory as empty."""
    calls: list[list[str]] = []

    def fake_run_oc(args: list[str]) -> subprocess.CompletedProcess:
        calls.append(args)
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(cluster, "run_oc", fake_run_oc)

    assert cluster.list_path("service-pod", "/app-root/ols-user-data/feedback") == []
    assert calls == [
        [
            "rsh",
            "service-pod",
            "sh",
            "-c",
            'if [ -e "$1" ] || [ -L "$1" ]; then ls "$1"; fi',
            "--",
            "/app-root/ols-user-data/feedback",
        ]
    ]
