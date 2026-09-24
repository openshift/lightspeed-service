"""Unit tests for E2E operator bundle installation helpers."""

from unittest.mock import Mock

import pytest

from tests.e2e.utils import ols_installer


def test_run_operator_sdk_bundle_has_process_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound bundle installation even when operator-sdk does not exit."""
    run = Mock()
    monkeypatch.setattr(ols_installer.subprocess, "run", run)

    ols_installer._run_operator_sdk_bundle("quay.io/example/bundle:latest")

    assert (
        run.call_args.kwargs["timeout"]
        == ols_installer.BUNDLE_INSTALL_SUBPROCESS_TIMEOUT
    )


def test_cleanup_failed_bundle_has_process_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound cleanup even when operator-sdk does not exit."""
    run = Mock()
    run.return_value.returncode = 0
    monkeypatch.setattr(ols_installer.subprocess, "run", run)

    assert ols_installer._cleanup_failed_bundle()
    assert (
        run.call_args.kwargs["timeout"]
        == ols_installer.BUNDLE_CLEANUP_SUBPROCESS_TIMEOUT
    )
