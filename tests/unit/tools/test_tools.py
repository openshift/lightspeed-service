"""Unit tests for tools module."""

from unittest.mock import patch

import pytest

from ols.src.tools.tools import (
    SENSITIVE_KEYWORDS,
    execute_tool_call,
    execute_tool_calls,
    get_tool_by_name,
    raise_for_sensitive_tool_args,
)


class FakeTool:
    """Mock tool class."""

    def __init__(self, name: str):
        """Initialize the tool with a name."""
        self.name = name

    async def arun(self, args):
        """Mock async run method."""
        return "fake_output"


def test_get_tool_by_name():
    """Test get_tool_by_name function."""
    fake_tool_name = "fake_tool"
    fake_tools = [FakeTool(name="fake_tool")]
    fake_tools_duplicite = [FakeTool(name="fake_tool"), FakeTool(name="fake_tool")]

    tool = get_tool_by_name(fake_tool_name, fake_tools)
    assert tool.name == fake_tool_name

    with pytest.raises(ValueError, match="Tool 'non_existent_tool' not found."):
        get_tool_by_name("non_existent_tool", fake_tools)

    with pytest.raises(ValueError, match="Multiple tools found with name 'fake_tool'."):
        get_tool_by_name(fake_tool_name, fake_tools_duplicite)


@pytest.mark.asyncio
async def test_execute_tool_call():
    """Test execute_tool_call function."""
    fake_tool_name = "fake_tool"
    fake_tool_args = {"arg1": "value1"}
    fake_tools = [FakeTool(name="fake_tool")]

    with patch(
        "ols.src.tools.tools.get_tool_by_name", return_value=FakeTool(fake_tool_name)
    ):
        status, output = await execute_tool_call(
            fake_tool_name, fake_tool_args, fake_tools
        )
        assert output == "fake_output"
        assert status == "success"

    with patch(
        "ols.src.tools.tools.get_tool_by_name", side_effect=Exception("Tool error")
    ):
        status, output = await execute_tool_call(
            fake_tool_name, fake_tool_args, fake_tools
        )
        assert "Error executing tool" in output
        assert status == "error"


@pytest.mark.asyncio
async def test_execute_tool_calls():
    """Test execute_tool_calls function."""
    fake_tool_name = "fake_tool"
    fake_tool_args = {"arg1": "value1"}
    fake_tools = [FakeTool(name="fake_tool")]

    tool_calls = [
        {"name": fake_tool_name, "args": fake_tool_args, "id": "tool_call_1"},
        {"name": fake_tool_name, "args": fake_tool_args, "id": "tool_call_2"},
    ]

    with patch(
        "ols.src.tools.tools.execute_tool_call", return_value=("success", "fake_output")
    ):
        tool_messages = await execute_tool_calls(tool_calls, fake_tools)
        assert len(tool_messages) == 2
        assert tool_messages[0].content == "fake_output"
        assert tool_messages[0].status == "success"
        assert tool_messages[0].tool_call_id == "tool_call_1"
        assert tool_messages[1].content == "fake_output"
        assert tool_messages[1].status == "success"
        assert tool_messages[1].tool_call_id == "tool_call_2"


def test_raise_for_sensitive_tool_args():
    """Test raise_for_sensitive_tool_args function."""
    raise_for_sensitive_tool_args({"tool_args": "normal_args"})

    for keyword in SENSITIVE_KEYWORDS:
        with pytest.raises(ValueError, match="Sensitive keyword in tool arguments"):
            sensitive_args = {"tool_args": keyword}
            raise_for_sensitive_tool_args(sensitive_args)


@pytest.mark.asyncio
async def test_execute_sensitive_tool_calls():
    """Test execute_tool_calls with sensitive tool arguments."""
    tool_calls = [
        {
            "name": "some_tool",
            "args": {"tool_arg": SENSITIVE_KEYWORDS[0]},
            "id": "tool_call_1",
        },
    ]
    fake_tools = []

    tool_messages = await execute_tool_calls(tool_calls, fake_tools)

    assert tool_messages[0].status == "error"
    assert "Sensitive keyword" in tool_messages[0].content
    assert "are not allowed" in tool_messages[0].content
