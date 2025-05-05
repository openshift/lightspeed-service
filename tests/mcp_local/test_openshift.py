"""Test the mcp_local.openshift module."""

import os
import pathlib
import subprocess
from unittest.mock import patch

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

from mcp_local.openshift import (
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
    redact_token,
    resolve_status_and_response,
    run_oc,
    safe_run_oc,
    show_pods_resource_usage,
    strip_args_for_oc_command,
)


@pytest.fixture(scope="function")
def token_in_env():
    """Set up a token in the environment for testing."""
    with patch.dict(
        os.environ,
        {"OC_USER_TOKEN": "fake-token"},
    ):
        yield


def test_strip_args_for_oc_command():
    """Test the strip_args_for_oc_command function."""
    # normal case
    args = ["pod", "my-pod"]
    expected = ["pod", "my-pod"]
    assert strip_args_for_oc_command(args) == expected

    # extra commands
    args = ["get", "pod", "my-pod"]
    expected = ["pod", "my-pod"]
    assert strip_args_for_oc_command(args) == expected

    # multiple extra commands
    args = ["oc", "get", "pod", "my-pod"]
    expected = ["pod", "my-pod"]
    assert strip_args_for_oc_command(args) == expected

    # extra spaces
    args = ["oc", " get ", " pod ", " my-pod "]
    expected = ["pod", "my-pod"]
    assert strip_args_for_oc_command(args) == expected

    # two commands as one
    args = ["pod my-pod"]
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
    status, result = safe_run_oc("get", args)
    assert status == "error"
    assert result == SECRET_NOT_ALLOWED_MSG

    # forbidden characters present
    args = ["pod", "my-pod;"]
    status, result = safe_run_oc("get", args)
    assert status == "error"
    assert result == BLOCKED_CHARS_DETECTED_MSG

    # normal case
    args = ["pod", "my-pod"]
    mocked_oc = "success", "stdout"
    with patch("mcp_local.openshift.run_oc", return_value=mocked_oc):
        status, response = safe_run_oc("get", args)
        assert status == "success"
        assert response == "stdout"


def test_token_default_value():
    """Test the default value of the token in the environment."""
    with patch("mcp_local.openshift.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="stdout and fake-token",
            stderr="",
        )
        run_oc([])

        assert "token-not-set" in mock_run.call_args[0][0]


def test_oc_run(token_in_env):
    """Test the run_oc function."""
    # normal case - token is in response - should be redacted
    args = ["pod", "my-pod"]
    with patch("mcp_local.openshift.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="stdout and fake-token",
            stderr="",
        )
        status, response = run_oc(args)

        # called with args and token
        expected_args = ["oc", *args, "--token", "fake-token"]
        assert expected_args == mock_run.call_args[0][0]

        assert status == "success"
        assert response == "stdout and <redacted>"

    # error case
    with patch("mcp_local.openshift.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr="stderr and fake-token",
        )
        status, response = run_oc(args)

        # called with args and token
        expected_args = ["oc", *args, "--token", "fake-token"]
        assert expected_args == mock_run.call_args[0][0]

        assert status == "error"
        assert response == "stderr and <redacted>"


def test_oc_run_exception(token_in_env):
    """Test the run_oc function on exception."""
    args = ["pod", "my-pod"]
    with patch("mcp_local.openshift.subprocess.run") as mock_run:
        mock_run.side_effect = Exception("error and fake-token")
        status, response = run_oc(args)

        # called with args and token
        expected_args = ["oc", *args, "--token", "fake-token"]
        assert expected_args == mock_run.call_args[0][0]

        assert status == "error"
        assert response.startswith("Error executing args")
        assert "Traceback" in response
        assert "<redacted>" in response
        assert "fake-token" not in response


@pytest.mark.parametrize("tool", (oc_get, oc_describe, oc_logs, oc_status, oc_adm_top))
def test_tools(tool, token_in_env):
    """Test tools that take arguments."""
    with patch("mcp_local.openshift.subprocess.run") as mock_run:
        args = ["irelevant"]
        mock_run.return_value = subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="stdout",
            stderr="",
        )

        status, result = tool(args)
        assert status == "success"
        assert result == "stdout"


def test_argless_tools(token_in_env):
    """Test tools that don't take any arguments."""
    with patch("mcp_local.openshift.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="stdout",
            stderr="",
        )

        status, result = show_pods_resource_usage()
        assert status == "success"
        assert result == "stdout"


@pytest.mark.asyncio
async def test_is_stdio_server():
    """Test if the server is a stdio server."""
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [
                    (
                        pathlib.Path(__file__).parent.parent.parent
                        / "mcp_local/openshift.py"
                    ).as_posix()
                ],
                "transport": "stdio",
            },
        }
    ) as client:
        tools = client.get_tools()
        assert len(tools) == 6
        assert tools[0].name == "oc_get"
        assert tools[1].name == "oc_describe"
        assert tools[2].name == "oc_logs"
        assert tools[3].name == "oc_status"
        assert tools[4].name == "show_pods_resource_usage"
        assert tools[5].name == "oc_adm_top"


def test_redact_token():
    """Test the redact_token function."""
    text = "texty text with token"
    token = "token"  # noqa: S105

    assert redact_token(text, token) == "texty text with <redacted>"
