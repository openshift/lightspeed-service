"""Test cases for the oc_cli module."""

import pytest

from ols.src.tools.oc_cli import sanitize_oc_args


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
