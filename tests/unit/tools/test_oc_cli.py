"""Test cases for the oc_cli module."""

from unittest.mock import MagicMock, patch

import pytest

from ols.src.tools.oc_cli import sanitize_oc_args, stdout_or_stderr, token_works_for_oc


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


def test_token_works_for_oc_failure():
    """Test token_works_for_oc failure scenario."""
    with patch("ols.src.tools.oc_cli.run_oc") as mock_run_oc:
        mock_process = MagicMock()
        mock_process.returncode = 1  # simulate failure
        mock_run_oc.return_value = mock_process
        assert not token_works_for_oc("some-token")


def test_token_works_for_oc_success():
    """Test token_works_for_oc success scenario."""
    with patch("ols.src.tools.oc_cli.run_oc") as mock_run_oc:
        mock_process = MagicMock()
        mock_process.returncode = 0  # simulate success
        mock_run_oc.return_value = mock_process
        assert token_works_for_oc("some-token")


def test_stdout_or_stderr():
    """Test stdout_or_stderr function."""
    mock_process = MagicMock()
    mock_process.stdout = "stdout"
    mock_process.stderr = "stderr"

    mock_process.returncode = 0
    assert "stdout" == stdout_or_stderr(mock_process)

    mock_process.returncode = 1
    assert "stderr" == stdout_or_stderr(mock_process)
