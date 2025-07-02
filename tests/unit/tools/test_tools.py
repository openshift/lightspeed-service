"""Unit tests for tools module."""

from unittest.mock import patch

import pytest
from langchain_core.tools.structured import StructuredTool

from ols.src.tools.tools import (
    execute_tool_call,
    execute_tool_calls,
    filter_read_only_tools,
    get_tool_by_name,
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


def test_filter_read_only_tools():
    """Test filter_read_only_tools function."""
    fake_tools = [
        StructuredTool(
            name="read_tool", args_schema={}, metadata={"readOnlyHint": True}
        ),
        StructuredTool(
            name="write_tool", args_schema={}, metadata={"readOnlyHint": False}
        ),
    ]

    read_only_tools = filter_read_only_tools(fake_tools)

    assert len(read_only_tools) == 1
    assert read_only_tools[0].name == "read_tool"


def test_filter_read_only_tools_no_metadata_tool(caplog):
    """Test filter_read_only_tools function."""
    fake_tools = [
        StructuredTool(name="no_metadata_tool", args_schema={}),
    ]

    read_only_tools = filter_read_only_tools(fake_tools)

    assert len(read_only_tools) == 1
    assert read_only_tools[0].name == "no_metadata_tool"

    assert "Tool no_metadata_tool has no metadata" in caplog.text
