"""Unit tests for tools module."""

import asyncio
import time
from typing import Optional
from unittest.mock import patch

import pytest
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel

from ols.src.tools.tools import (
    SENSITIVE_KEYWORDS,
    execute_tool_call,
    execute_tool_calls,
    get_tool_by_name,
    raise_for_sensitive_tool_args,
)


class FakeSchema(BaseModel):
    """Fake schema for FakeTool."""


class FakeTool(StructuredTool):
    """Mock tool class that inherits from StructuredTool."""

    def __init__(self, name: str, delay: float = 0.0, should_fail: bool = False):
        """Initialize the tool with a name."""
        # Initialize StructuredTool with required parameters
        super().__init__(
            name=name,
            description=f"Fake tool {name}",
            func=self._sync_run,
            args_schema=FakeSchema,
        )
        self._delay = delay
        self._should_fail = should_fail

    def _sync_run(self, **kwargs) -> str:
        """Sync run method (required by StructuredTool)."""
        return f"fake_output_from_{self.name}"

    async def arun(self, tool_args: Optional[dict] = None, **kwargs) -> str:
        """Mock async run method."""
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._should_fail:
            raise Exception("Tool execution failed")
        return f"fake_output_from_{self.name}"


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
        assert output == "fake_output_from_fake_tool"
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
async def test_execute_tool_calls_empty():
    """Test execute_tool_calls with empty tool calls list."""
    tool_messages = await execute_tool_calls([], [])
    assert tool_messages == []


@pytest.mark.asyncio
async def test_execute_tool_calls_parallel_execution():
    """Test that tool calls are executed in parallel, not sequentially."""
    # Create tools with delays to test parallel execution
    fake_tools = [
        FakeTool(name="tool1", delay=0.1),
        FakeTool(name="tool2", delay=0.1),
        FakeTool(name="tool3", delay=0.1),
    ]

    tool_calls = [
        {"name": "tool1", "args": {}, "id": "call_1"},
        {"name": "tool2", "args": {}, "id": "call_2"},
        {"name": "tool3", "args": {}, "id": "call_3"},
    ]

    start_time = time.time()
    tool_messages = await execute_tool_calls(tool_calls, fake_tools)
    end_time = time.time()

    # If executed in parallel, total time should be close to 0.1s (the delay)
    # If executed sequentially, it would be close to 0.3s (3 * 0.1s)
    execution_time = end_time - start_time
    assert (
        execution_time < 0.25
    ), f"Execution took {execution_time}s, expected < 0.25s for parallel execution"

    # Check that all tools were executed
    assert len(tool_messages) == 3
    assert tool_messages[0].content == "fake_output_from_tool1"
    assert tool_messages[1].content == "fake_output_from_tool2"
    assert tool_messages[2].content == "fake_output_from_tool3"


@pytest.mark.asyncio
async def test_execute_tool_calls_with_missing_tool_name():
    """Test execute_tool_calls with missing tool name."""
    fake_tools = [FakeTool(name="fake_tool")]

    tool_calls = [
        {"name": None, "args": {}, "id": "call_1"},  # Missing tool name
        {"name": "fake_tool", "args": {}, "id": "call_2"},  # Valid tool call
    ]

    with patch(
        "ols.src.tools.tools.execute_tool_call", return_value=("success", "fake_output")
    ):
        tool_messages = await execute_tool_calls(tool_calls, fake_tools)
        assert len(tool_messages) == 2
        assert tool_messages[0].content == "Error: Tool name is missing from tool call"
        assert tool_messages[0].status == "error"
        assert tool_messages[0].tool_call_id == "call_1"
        assert tool_messages[1].content == "fake_output"
        assert tool_messages[1].status == "success"
        assert tool_messages[1].tool_call_id == "call_2"


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


@pytest.mark.asyncio
async def test_execute_tool_calls_mixed_success_and_failure():
    """Test execute_tool_calls with a mix of successful and failing tools."""
    fake_tools = [
        FakeTool(name="success_tool"),
        FakeTool(name="fail_tool", should_fail=True),
    ]

    tool_calls = [
        {"name": "success_tool", "args": {}, "id": "call_1"},
        {"name": "fail_tool", "args": {}, "id": "call_2"},
        {"name": "nonexistent_tool", "args": {}, "id": "call_3"},
    ]

    tool_messages = await execute_tool_calls(tool_calls, fake_tools)

    assert len(tool_messages) == 3

    # First call should succeed
    assert tool_messages[0].status == "success"
    assert tool_messages[0].content == "fake_output_from_success_tool"

    # Second call should fail due to tool raising exception
    assert tool_messages[1].status == "error"
    assert "Tool execution failed" in tool_messages[1].content

    # Third call should fail due to nonexistent tool
    assert tool_messages[2].status == "error"
    assert "Tool 'nonexistent_tool' not found" in tool_messages[2].content


@pytest.mark.asyncio
async def test_execute_tool_calls_preserves_tool_call_order():
    """Test that tool messages are returned in the same order as tool calls."""
    fake_tools = [
        FakeTool(name="tool_a"),
        FakeTool(name="tool_b"),
        FakeTool(name="tool_c"),
    ]

    tool_calls = [
        {"name": "tool_c", "args": {}, "id": "call_c"},
        {"name": "tool_a", "args": {}, "id": "call_a"},
        {"name": "tool_b", "args": {}, "id": "call_b"},
    ]

    tool_messages = await execute_tool_calls(tool_calls, fake_tools)

    assert len(tool_messages) == 3
    # Order should match input order, not alphabetical
    assert tool_messages[0].tool_call_id == "call_c"
    assert tool_messages[0].content == "fake_output_from_tool_c"
    assert tool_messages[1].tool_call_id == "call_a"
    assert tool_messages[1].content == "fake_output_from_tool_a"
    assert tool_messages[2].tool_call_id == "call_b"
    assert tool_messages[2].content == "fake_output_from_tool_b"
