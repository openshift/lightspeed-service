"""Test cases for the oc_cli module."""

import subprocess
from unittest.mock import patch

import pytest

from ols.src.tools.oc_cli import sanitize_oc_args, token_works_for_oc


@pytest.mark.parametrize(
    "input_args, expected_output",
    [
        (["get", "pods"], ["pods"]),
        (["oc", "adm", "top", "pods", "-A"], ["pods", "-A"]),
        (["get", "pods; rm -rf /"], ["pods"]),
        (["get", "pods & rm -rf /"], ["pods"]),
        (["get", "pods | cat /etc/passwd"], ["pods"]),
        (["get", "pods `ls`"], ["pods"]),
        (["get", "pods $(rm -rf /)"], ["pods"]),
        (["get", "pods \\& whoami"], ["pods"]),
        (["get", "pods; rm -rf / & echo hacked | cat /etc/passwd"], ["pods"]),
    ],
)
def test_sanitize_oc_args(input_args, expected_output):
    """Test that oc args are sanitized."""
    assert sanitize_oc_args(input_args) == expected_output


def test_token_works_for_oc_failure():
    """Test token_works_for_oc failure scenario."""
    with patch(
        "ols.src.tools.oc_cli.run_oc",
        side_effect=subprocess.CalledProcessError(1, "oc"),
    ):
        assert not token_works_for_oc("some-token")


def test_token_works_for_oc_success():
    """Test token_works_for_oc success scenario."""
    with patch(
        "ols.src.tools.oc_cli.run_oc",
        returns_value=subprocess.CompletedProcess("oc", 0),
    ):
        assert token_works_for_oc("some-token")
