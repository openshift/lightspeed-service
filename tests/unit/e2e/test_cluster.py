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


def test_run_oc_passes_custom_timeout_to_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow callers to extend the default oc command timeout."""
    calls: list[dict[str, object]] = []

    def fake_subprocess_run(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(cluster.subprocess, "run", fake_subprocess_run)

    cluster.run_oc(["get", "pods"], timeout=600)

    assert calls[0]["timeout"] == 600
