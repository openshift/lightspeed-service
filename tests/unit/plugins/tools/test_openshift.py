"""Test cases for the oc_cli module."""

from unittest.mock import PropertyMock, patch

import pytest
from langchain.tools import tool
from langchain_core.tools.base import BaseTool

from ols.plugins.tools.openshift import OCToolProvider, sanitize_oc_args
from ols.plugins.tools.tools import ToolProvidersRegistry, ToolsContext


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


def test_openshift_tool_provider_is_registered():
    """Test that the openshift tool provider is registered."""
    assert "openshift" in ToolProvidersRegistry._all_tool_providers
    assert isinstance(
        ToolProvidersRegistry._all_tool_providers["openshift"], OCToolProvider
    )


def test_openshift_tool_provider_tools():
    """Test that the openshift tool provider has the expected tools."""
    tools = OCToolProvider().tools
    assert isinstance(tools, dict)
    assert len(tools)
    assert all(isinstance(tool, BaseTool) for tool in tools.values())


def test_execute_happy_tool(caplog):
    """Test that the openshift tool provider can execute happy tools."""
    caplog.set_level(10)  # set debug level

    @tool
    def fake_tool(fake_arg: str):
        """Fake tool for testing."""
        return "fake_tool_response"

    with patch.object(OCToolProvider, "tools", new_callable=PropertyMock) as mock_prop:
        mock_prop.return_value = {"fake_tool": fake_tool}

        tool_provider = OCToolProvider()
        result = tool_provider.execute_tool(
            "fake_tool",
            {"fake_arg": "hello"},
            ToolsContext(user_token="fake-token"),  # noqa: S106
        )
        assert result == ("fake_tool_response", "success")

    # ensure token is not leaked into logs
    assert "fake-token" not in caplog.text


def test_execute_sad_tool(caplog):
    """Test that the openshift tool provider can execute sad tools."""
    caplog.set_level(10)  # set debug level

    @tool
    def fake_tool(fake_arg: str):
        """Fake tool for testing."""
        raise ValueError("I'm sad")

    with patch.object(OCToolProvider, "tools", new_callable=PropertyMock) as mock_prop:
        mock_prop.return_value = {"fake_tool": fake_tool}

        tool_provider = OCToolProvider()
        result = tool_provider.execute_tool(
            "fake_tool",
            {"fake_arg": "hello"},
            ToolsContext(user_token="fake-token"),  # noqa: S106
        )
        assert result == ("Error: I'm sad", "error")

    # ensure token is not leaked into logs
    assert "fake-token" not in caplog.text
