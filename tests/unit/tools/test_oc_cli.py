"""Test cases for the oc_cli module."""

import os
import subprocess
from unittest.mock import patch

import pytest

from ols.src.tools.oc_cli import log_to_oc, sanitize_oc_args


@pytest.mark.parametrize(
    "input_args, expected_output",
    [
        (["get", "pods"], ["get", "pods"]),
        (["get", "pods; rm -rf /"], ["get", "pods"]),
        (["get", "pods & rm -rf /"], ["get", "pods"]),
        (["get", "pods | cat /etc/passwd"], ["get", "pods"]),
        (["get", "pods `ls`"], ["get", "pods"]),
        (["get", "pods $(rm -rf /)"], ["get", "pods"]),
        (["get", "pods \\& whoami"], ["get", "pods"]),
        (["get", "pods; rm -rf / & echo hacked | cat /etc/passwd"], ["get", "pods"]),
    ],
)
def test_sanitize_oc_args(input_args, expected_output):
    """Test that oc args are sanitized."""
    assert sanitize_oc_args(input_args) == expected_output


def test_log_to_oc_missing_server():
    """Test log to oc."""
    with patch("ols.src.tools.oc_cli.run_oc"):

        # send None as a server
        assert not log_to_oc("some-token", None)

        # env KUBERNETES_SERVICE_HOST is not set - empty env
        with patch.dict(os.environ, {}):
            assert not log_to_oc("some-token", None)

        # env KUBERNETES_SERVICE_HOST is empty string
        with patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": ""}):
            assert not log_to_oc("some-token", None)


def test_log_to_oc_success():
    """Test log to oc."""
    with patch("ols.src.tools.oc_cli.run_oc"):
        assert log_to_oc("some-token", "some-server")


def test_log_to_oc_error():
    """Test log to oc."""
    # invalid token error
    with patch(
        "ols.src.tools.oc_cli.run_oc",
        side_effect=subprocess.CalledProcessError(1, "", stderr="token invalid"),
    ):
        assert not log_to_oc("some-token", "some-server")

    # other error
    with patch(
        "ols.src.tools.oc_cli.run_oc",
        side_effect=subprocess.CalledProcessError(
            1, "", stderr="something else went wrong"
        ),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            log_to_oc("some-token", "some-server")
