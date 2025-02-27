"""Unit tests for tools module."""

import pytest
from langchain.tools import tool

from ols.src.tools.tools import check_tool_description_length, execute_tool_calls


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
    def tool1(value: str):
        """Tool 1."""
        return f"Tool 1: {value}"

    @tool
    def tool2(value: str):
        """Tool 2."""
        raise ValueError("Tool 2 error")

    tools_map = {"tool1": tool1, "tool2": tool2}
    tool_calls = [
        {"id": 1, "name": "tool1", "args": {"value": "args1"}},
        {"id": 2, "name": "tool2", "args": {"value": "args2"}},
    ]

    tool_messages = execute_tool_calls(tools_map, tool_calls)
    assert len(tool_messages) == 2
    assert tool_messages[0].content == "Tool 1: args1"
    assert tool_messages[1].content == "Error executing tool2: Tool 2 error"
