"""Test cases for the oc_cli module."""

from unittest.mock import MagicMock, patch

import pytest

from ols.src.tools.oc_cli import (
    BLOCKED_CHARS,
    BLOCKED_CHARS_DETECTED_MSG,
    SECRET_NOT_ALLOWED_MSG,
    is_blocked_char_in_args,
    is_secret_in_args,
    safe_run_oc,
    stdout_or_stderr,
    strip_args_for_oc_command,
)


@pytest.mark.parametrize(
    "input_args, expected_output",
    [
        (["oc", "get", "pods"], ["pods"]),
        (["get", "pods"], ["pods"]),
        (["oc", "adm", "top", "pods", "-A"], ["pods", "-A"]),
        (["describe", "pods"], ["pods"]),
        (["logs", "pod-name"], ["pod-name"]),
        (["status", "pod-name"], ["pod-name"]),
        (["oc get pods"], ["pods"]),  # single string
    ],
)
def test_strip_args_for_oc_command(input_args, expected_output):
    """Test that oc args are sanitized."""
    assert strip_args_for_oc_command(input_args) == expected_output


def test_is_blocked_char_in_args():
    """Test that blocked characters are detected."""
    for char in BLOCKED_CHARS:
        assert is_blocked_char_in_args(["oc", "get", "pods", char]) is True

    assert is_blocked_char_in_args(["oc", "get", "pods"]) is False


def test_is_secret_in_args():
    """Test that secret characters are detected."""
    assert is_secret_in_args(["oc", "get", "pods", "secret"]) is True
    assert is_secret_in_args(["oc", "get", "pods", "secrets"]) is True
    assert is_secret_in_args(["oc", "get", "pods"]) is False


def test_stdout_or_stderr():
    """Test stdout_or_stderr function."""
    mock_process = MagicMock()
    mock_process.stdout = "stdout"
    mock_process.stderr = "stderr"

    assert "stdout" == stdout_or_stderr(mock_process)

    mock_process.stdout = ""
    assert "stderr" == stdout_or_stderr(mock_process)


def test_safe_run_oc():
    """Test safe_run_oc function."""
    with patch("ols.src.tools.oc_cli.run_oc") as mock_run_oc:
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "output"
        mock_run_oc.return_value = mock_process

        result = safe_run_oc(["get", "pods", BLOCKED_CHARS[0]], "some-token")
        assert result == BLOCKED_CHARS_DETECTED_MSG

        result = safe_run_oc(["oc", "get", "secret"], "some-token")
        assert result == SECRET_NOT_ALLOWED_MSG

        result = safe_run_oc(["get", "pods"], "some-token")
        assert result == "output"

        mock_process.stdout = ""
        mock_process.stderr = "error"
        result = safe_run_oc(["get", "pods"], "some-token")
        assert result == "error"
