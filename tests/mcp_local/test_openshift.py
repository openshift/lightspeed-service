"""Test the mcp_local.openshift module."""

import os
import pathlib
import re
import subprocess
from unittest.mock import patch

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

from mcp_local.openshift import (
    BLOCKED_CHARS,
    BLOCKED_CHARS_DETECTED_MSG,
    SECRET_NOT_ALLOWED_MSG,
    oc_adm_top,
    oc_describe,
    oc_get,
    oc_logs,
    oc_status,
    raise_for_unacceptable_args,
    redact_token,
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


def test_raise_for_unacceptable_args():
    """Test the unacceptable args check."""
    # blocked character present
    for char in BLOCKED_CHARS:
        args = ["oc", "get", f"pod{char}my-pod"]
        with pytest.raises(Exception, match=re.escape(BLOCKED_CHARS_DETECTED_MSG)):
            raise_for_unacceptable_args(args)

    # secret/secrets present
    for s in ["secret", "secrets"]:
        args = ["oc", "get", s, "my-secret"]
        with pytest.raises(Exception, match=SECRET_NOT_ALLOWED_MSG):
            raise_for_unacceptable_args(args)

    # no blocked character, no error (returns nothing)
    assert raise_for_unacceptable_args(["oc", "get", "pod", "my-pod"]) is None

    # empty list, no error (returns nothing)
    assert raise_for_unacceptable_args([]) is None


def test_safe_run_oc():
    """Test the run_oc function."""
    # secret present
    with pytest.raises(Exception, match=SECRET_NOT_ALLOWED_MSG):
        safe_run_oc("get", ["secret"])

    # forbidden characters present
    with pytest.raises(Exception, match=re.escape(BLOCKED_CHARS_DETECTED_MSG)):
        safe_run_oc("get", ["pod", "my-pod;"])

    # normal case
    args = ["pod", "my-pod"]
    mocked_oc = "stdout"
    with patch("mcp_local.openshift.run_oc", return_value=mocked_oc):
        response = safe_run_oc("get", args)
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
    args = ["pod", "my-pod"]
    expected_args = ["oc", *args, "--token", "fake-token"]

    # normal case - token is in response - should be redacted
    with patch("mcp_local.openshift.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="stdout and fake-token",
            stderr="",
        )
        response = run_oc(args)

        # called with args and token
        assert expected_args == mock_run.call_args[0][0]

        assert response == "stdout and <redacted>"

    # error case
    with patch("mcp_local.openshift.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr="stderr and fake-token",
        )
        with pytest.raises(Exception, match="stderr and <redacted>"):
            run_oc(args)

    # quasi case
    with patch("mcp_local.openshift.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["oc", "get", "pod", "my-pod"],
            returncode=0,
            stdout="",
            stderr="stderr and fake-token",
        )
        response = run_oc(args)

        # called with args and token
        assert expected_args == mock_run.call_args[0][0]

        assert response == "stderr and <redacted>"

    # exception case
    with patch("mcp_local.openshift.subprocess.run") as mock_run:
        mock_run.side_effect = Exception("error and fake-token")

        with pytest.raises(Exception) as exception:
            run_oc(args)
        assert "Traceback" in str(exception.value)
        assert "error and <redacted>" in str(exception.value)
        assert "fake-token" not in str(exception.value)


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

        result = tool(args)
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

        result = show_pods_resource_usage()
        assert result == "stdout"


@pytest.mark.asyncio
async def test_is_stdio_server():
    """Test if the server is a stdio server."""
    mcp_client = MultiServerMCPClient(
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
    )

    tools = await mcp_client.get_tools()
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
