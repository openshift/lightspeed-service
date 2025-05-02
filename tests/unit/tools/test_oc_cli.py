"""Test the mcp_local.openshift module."""

import subprocess
from unittest.mock import patch

import pytest

from ols.src.tools.oc_cli import (
    BLOCKED_CHARS,
    BLOCKED_CHARS_DETECTED_MSG,
    SECRET_NOT_ALLOWED_MSG,
    is_blocked_char_in_args,
    is_secret_in_args,
    oc_adm_top,
    oc_describe,
    oc_get,
    oc_logs,
    oc_status,
    resolve_status_and_response,
    run_oc,
    safe_run_oc,
    show_pods_resource_usage,
    strip_args_for_oc_command,
)


def test_strip_args_for_oc_command():
    """Test the strip_args_for_oc_command function."""
    # normal case
    args = ["pod", "my-pod"]
    expected = ["pod", "my-pod"]
    assert strip_args_for_oc_command(args) == expected

    # extra commands
    args = ["oc", "get", "pod", "my-pod"]
    expected = ["pod", "my-pod"]
    assert strip_args_for_oc_command(args) == expected

    # extra spaces
    args = ["oc", " get ", " pod ", " my-pod "]
    expected = ["pod", "my-pod"]
    assert strip_args_for_oc_command(args) == expected

    # no command
    args = ["get", "pod", "my-pod"]
    expected = ["pod", "my-pod"]
    assert strip_args_for_oc_command(args) == expected

    # empty list
    args = []
    expected = []
    assert strip_args_for_oc_command(args) == expected


def test_is_blocked_char_in_args():
    """Test the is_blocked_char_in_args function."""
    # blocked character present
    for char in BLOCKED_CHARS:
        args = ["oc", "get", f"pod{char}my-pod"]
        assert is_blocked_char_in_args(args) is True

    # no blocked character
    args = ["oc", "get", "pod", "my-pod"]
    assert is_blocked_char_in_args(args) is False

    # empty list
    args = []
    assert is_blocked_char_in_args(args) is False


def test_is_secret_in_args():
    """Test the is_secret_in_args function."""
    # secret present
    args = ["oc", "get", "secret", "my-secret"]
    assert is_secret_in_args(args) is True

    # secrets present
    args = ["oc", "get", "secrets", "my-secret"]
    assert is_secret_in_args(args) is True

    # no secret
    args = ["oc", "get", "pod", "my-pod"]
    assert is_secret_in_args(args) is False

    # empty list
    args = []
    assert is_secret_in_args(args) is False


def test_resolve_status_and_response():
    """Test the resolve_status_and_response function."""
    # normal case
    result = subprocess.CompletedProcess(
        args=["oc", "get", "pod", "my-pod"],
        returncode=0,
        stdout="stdout",
        stderr="",
    )
    status, response = resolve_status_and_response(result)
    assert status == "success"
    assert response == "stdout"

    # quasi case
    result = subprocess.CompletedProcess(
        args=["oc", "get", "pod", "my-pod"],
        returncode=0,
        stdout="",
        stderr="stderr",
    )
    status, response = resolve_status_and_response(result)
    assert status == "success"
    assert response == "stderr"

    # error case
    result = subprocess.CompletedProcess(
        args=["oc", "get", "pod", "my-pod"],
        returncode=1,
        stdout="",
        stderr="stderr",
    )
    status, response = resolve_status_and_response(result)
    assert status == "error"
    assert response == "stderr"


def test_safe_run_oc():
    """Test the run_oc function."""
    # secret present
    args = ["secret"]
    status, result = safe_run_oc("get", args, "fake-token")
    assert status == "error"
    assert result == SECRET_NOT_ALLOWED_MSG

    # forbidden characters present
    args = ["pod", "my-pod;"]
    status, result = safe_run_oc("get", args, "fake-token")
    assert status == "error"
    assert result == BLOCKED_CHARS_DETECTED_MSG

    # normal case
    args = ["pod", "my-pod"]
    mocked_oc = "success", "stdout"
    with patch("ols.src.tools.oc_cli.run_oc", return_value=mocked_oc):
        status, response = safe_run_oc("get", args, "fake-token")
        assert status == "success"
        assert response == "stdout"


def test_oc_run():
    """Test the run_oc function."""
    # normal case - token is in response - should be redacted
    args = ["pod", "my-pod"]
    with patch("ols.src.tools.oc_cli.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="stdout and fake-token",
            stderr="",
        )
        status, response = run_oc(args, "fake-token")

        # called with args and token
        expected_args = ["oc", *args, "--token", "fake-token"]
        assert expected_args == mock_run.call_args[0][0]

        assert status == "success"
        assert response == "stdout and <redacted>"

    # error case
    with patch("ols.src.tools.oc_cli.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr="stderr and fake-token",
        )
        status, response = run_oc(args, "fake-token")

        # called with args and token
        expected_args = ["oc", *args, "--token", "fake-token"]
        assert expected_args == mock_run.call_args[0][0]

        assert status == "error"
        assert response == "stderr and <redacted>"


def test_oc_run_exception():
    """Test the run_oc function on exception."""
    args = ["pod", "my-pod"]
    with patch("ols.src.tools.oc_cli.subprocess.run") as mock_run:
        mock_run.side_effect = Exception("error and fake-token")
        status, response = run_oc(args, "fake-token")

        # called with args and token
        expected_args = ["oc", *args, "--token", "fake-token"]
        assert expected_args == mock_run.call_args[0][0]

        assert status == "error"
        assert response.startswith("Error executing args")
        assert "Traceback" in response
        assert "<redacted>" in response
        assert "fake-token" not in response


@pytest.mark.parametrize(
    "tool, arg_name",
    (
        (oc_get, "oc_get_args"),
        (oc_describe, "oc_describe_args"),
        (oc_logs, "oc_logs_args"),
        (oc_status, "oc_status_args"),
        (oc_adm_top, "oc_adm_top_args"),
    ),
)
def test_tools(tool, arg_name):
    """Test tools that take arguments."""
    with patch("ols.src.tools.oc_cli.subprocess.run") as mock_run:
        args = ["irelevant"]
        mock_run.return_value = subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="stdout",
            stderr="",
        )

        status, result = tool.invoke({arg_name: args, "token": "fake-token"})
        assert status == "success"
        assert result == "stdout"


def test_argless_tools():
    """Test tools that don't take any arguments."""
    with patch("ols.src.tools.oc_cli.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="stdout",
            stderr="",
        )

        status, result = show_pods_resource_usage.invoke({"token": "fake-token"})
        assert status == "success"
        assert result == "stdout"
