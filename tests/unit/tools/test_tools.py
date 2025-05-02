"""Unit tests for tools module."""

import pytest
from langchain.tools import tool

from ols.src.tools.tools import (
    check_tool_description_length,
    execute_oc_tool_calls,
    get_available_tools,
    oc_tools,
)


def test_check_tool_description_length():
    """Test check_tool_description_length."""

    @tool
    def ok_tool():
        """Ok description."""

    check_tool_description_length(ok_tool)

    @tool
    def bad_tool():
        """Too long description."""

    bad_tool.description = "a" * 1024

    with pytest.raises(ValueError):
        check_tool_description_length(bad_tool)


def test_execute_tool_calls():
    """Test execute_tool_calls."""

    @tool
    def tool1(command_args: list[str]):
        """Tool 1."""
        return "success", "Tool 1 called"

    @tool
    def tool2(command_args: list[str]):
        """Tool 2."""
        raise ValueError("some error")

    tools_map = {"tool1": tool1, "tool2": tool2}
    tool_calls = [
        {"id": 1, "name": "tool1", "args": {"command_args": ["args1"]}},
        {"id": 2, "name": "tool2", "args": {"command_args": ["args2"]}},
    ]

    tool_messages = execute_oc_tool_calls(tools_map, tool_calls, "fake-token")

    assert tool_messages[0].content == "Tool 1 called"
    assert tool_messages[0].tool_call_id == "1"
    assert tool_messages[0].status == "success"
    assert tool_messages[1].content.startswith("Error executing tool 'tool2'")
    assert tool_messages[1].tool_call_id == "2"
    assert tool_messages[1].status == "error"


def test_execute_oc_tool_calls_not_leaks_token_into_logs(caplog):
    """Test execute_oc_tool_calls does not leak token into logs."""
    caplog.set_level(10)  # set debug level

    @tool
    def tool1(some_arg: str):
        """Tool 1."""
        return "success", "Tool 1 called"

    tools_map = {"tool1": tool1}
    tool_calls = [{"id": 1, "name": "tool1", "args": {"some_arg": "blah"}}]

    execute_oc_tool_calls(tools_map, tool_calls, "fake-token")

    assert (
        "Tool: tool1 | Args: {'some_arg': 'blah'} | Output: Tool 1 called"
        in caplog.text
    )
    assert "fake-token" not in caplog.text


def test_execute_oc_tool_calls_not_leaks_token_into_output(caplog):
    """Test execute_oc_tool_calls does not leak token into output."""
    caplog.set_level(10)  # set debug level

    @tool
    def tool1(some_args: list):
        """Tool 1."""
        return "success", "bla"

    tools_map = {"tool1": tool1}

    # missing args
    tool_calls = [{"id": 1, "name": "tool1"}]
    tool_messages = execute_oc_tool_calls(tools_map, tool_calls, "fake-token")
    assert len(tool_messages) == 1
    assert "fake-token" not in tool_messages[0].content

    # unknown args
    tool_calls = [{"id": 1, "name": "tool1", "args": {"unknown_args": "blo"}}]
    tool_messages = execute_oc_tool_calls(tools_map, tool_calls, "fake-token")
    assert len(tool_messages) == 1
    assert "fake-token" not in tool_messages[0].content

    # ensure the token is also not in the logs
    assert "fake-token" not in caplog.text


def test_get_available_tools():
    """Test get_available_tools."""
    tools = get_available_tools(introspection_enabled=False)
    assert tools == {}

    tools = get_available_tools(introspection_enabled=True, user_token=None)
    assert tools == {}

    tools = get_available_tools(introspection_enabled=True, user_token="")
    assert tools == {}


def test_execute_oc_tool_calls_not_leaks_token_on_error(caplog):
    """Test execute_oc_tool_calls does not leak token into output."""
    caplog.set_level(10)  # set debug level

    @tool
    def tool1(some_arg: str, token: str):
        """Tool 1."""
        raise ValueError(some_arg, token)

    tools_map = {"tool1": tool1}

    # missing args
    tool_calls = [{"id": 1, "name": "tool1", "args": {"some_arg": "bla"}}]
    tool_messages = execute_oc_tool_calls(tools_map, tool_calls, "fake-token")
    assert len(tool_messages) == 1
    assert "fake-token" not in tool_messages[0].content
    assert "<redacted>" in tool_messages[0].content

    # ensure the token is also not in the logs
    assert "fake-token" not in caplog.text
    assert "<redacted>" in caplog.text
    tools = get_available_tools(
        introspection_enabled=True, user_token="token-value"  # noqa: S106
    )
    assert tools == oc_tools
