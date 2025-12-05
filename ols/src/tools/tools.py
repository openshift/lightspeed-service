"""Functions/Tools definition."""

import asyncio
import json
import logging
from typing import Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools.structured import StructuredTool

from ols.app.models.config import MCPServerConfig
from ols.app.models.masking_config import MaskingConfig
from ols.src.maskers.data_masking_service import DataMaskingService

logger = logging.getLogger(__name__)


SENSITIVE_KEYWORDS = ["secret"]

# Global data masking service instance
_data_masking_service: Optional[DataMaskingService] = None
_mcp_server_configs: dict[str, MCPServerConfig] = {}


def initialize_data_masking(mcp_servers: list[MCPServerConfig]) -> None:
    """Initialize the data masking service with MCP server configurations.

    This should be called during application startup to set up server-specific
    masking configurations.

    Args:
        mcp_servers: List of MCP server configurations
    """
    global _mcp_server_configs, _data_masking_service

    # Build server config lookup
    _mcp_server_configs = {server.name: server for server in mcp_servers}

    # Create masking service with config callback
    _data_masking_service = DataMaskingService(_get_server_masking_config)

    logger.info(
        "DataMaskingService initialized with %d server configurations",
        len(_mcp_server_configs),
    )


def _get_server_masking_config(server_name: str) -> Optional[MaskingConfig]:
    """Get masking configuration for a specific server.

    Args:
        server_name: Name of the MCP server

    Returns:
        The MaskingConfig for the server, or None if not configured
    """
    server_config = _mcp_server_configs.get(server_name)
    if server_config and server_config.data_masking:
        return server_config.data_masking
    return None


def get_data_masking_service(
    config_callback: Optional[callable] = None,
) -> DataMaskingService:
    """Get or create the global DataMaskingService instance.

    Args:
        config_callback: Optional callback to get server masking config.
            If None, the service will use a default security pattern group.

    Returns:
        The DataMaskingService instance
    """
    global _data_masking_service
    if _data_masking_service is None:
        _data_masking_service = DataMaskingService(config_callback)
    return _data_masking_service


def mask_tool_output(tool_output: str, server_name: Optional[str] = None) -> str:
    """Apply data masking to tool output.

    Args:
        tool_output: The raw output from an MCP tool
        server_name: Optional name of the MCP server for server-specific masking

    Returns:
        The masked output with sensitive data replaced
    """
    service = get_data_masking_service()

    # Apply masking using default security pattern group
    # This provides baseline protection even without server-specific config
    try:
        result_dict = {"result": tool_output}
        if server_name:
            # Try server-specific masking first
            masked_dict = service.mask_response(result_dict, server_name)
        else:
            # Fall back to default security patterns
            masked_dict = service.mask_data(result_dict, "security")
        return masked_dict.get("result", tool_output)
    except Exception as e:
        logger.error(f"Error during data masking: {e}")
        # On error, return masked placeholder for safety
        return "[REDACTED: masking error]"


def raise_for_sensitive_tool_args(tool_args: dict) -> None:
    """Check tool arguments for sensitive content and raise an exception if found."""
    for key, value in tool_args.items():
        if any(keyword in str(value).lower() for keyword in SENSITIVE_KEYWORDS):
            raise ValueError(
                f"Sensitive keyword in tool arguments {key}={value} are not allowed."
            )


def get_tool_by_name(
    tool_name: str, all_mcp_tools: list[StructuredTool]
) -> StructuredTool:
    """Get a tool by its name from the MCP client."""
    tool = [tool for tool in all_mcp_tools if tool.name == tool_name]
    if len(tool) == 0:
        raise ValueError(f"Tool '{tool_name}' not found.")
    if len(tool) > 1:
        # TODO: LCORE-94
        raise ValueError(f"Multiple tools found with name '{tool_name}'.")
    return tool[0]


async def execute_tool_call(
    tool_name: str, tool_args: dict, all_mcp_tools: list[StructuredTool]
) -> tuple[str, str]:
    """Execute a tool call and return the output and status.

    The output is automatically masked to prevent sensitive data leakage.
    """
    try:
        tool = get_tool_by_name(tool_name, all_mcp_tools)
        tool_output = await tool.arun(_jsonify(tool_args))  # type: ignore [attr-defined]

        # Apply data masking to the tool output
        masked_output = mask_tool_output(str(tool_output))

        status = "success"
        logger.debug(
            "Tool: %s | Args: %s | Output (masked): %s",
            tool_name,
            tool_args,
            masked_output[:500] if len(masked_output) > 500 else masked_output,
        )
        tool_output = masked_output
    except Exception as e:
        # catching generic exception here - if it contains something it
        # shouldn't (eg. token in openshift tools), it is responsibility
        # of the mcp/openshift tools to ensure nothing is leaked
        tool_output = f"Error executing tool '{tool_name}': {e}"
        status = "error"
        logger.exception(tool_output)
    return status, tool_output


async def _execute_single_tool_call(
    tool_call: dict, all_mcp_tools: list[StructuredTool]
) -> ToolMessage:
    """Execute a single tool call and return a ToolMessage."""
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})
    tool_id = tool_call.get("id")

    if tool_name is None:
        tool_output = "Error: Tool name is missing from tool call"
        status = "error"
        logger.error("Tool call missing name: %s", tool_call)
    else:
        try:
            raise_for_sensitive_tool_args(tool_args)
            status, tool_output = await execute_tool_call(
                tool_name, tool_args, all_mcp_tools
            )
        except Exception as e:
            tool_output = (
                f"Error executing tool '{tool_name}' with args {tool_args}: {e}"
            )
            status = "error"
            logger.exception(tool_output)

    return ToolMessage(content=tool_output, status=status, tool_call_id=tool_id)


async def execute_tool_calls(
    tool_calls: list[dict],
    all_mcp_tools: list[StructuredTool],
) -> list[ToolMessage]:
    """Execute tool calls in parallel and return ToolMessages."""
    if not tool_calls:
        return []

    # Create tasks for parallel execution
    tasks = [
        _execute_single_tool_call(tool_call, all_mcp_tools) for tool_call in tool_calls
    ]

    # Execute all tool calls in parallel
    tool_messages = await asyncio.gather(*tasks)

    return tool_messages


def _jsonify(args: dict) -> dict:
    """Convert to JSON."""
    res = {}
    for key, value in args.items():
        if isinstance(value, str) and _maybe_json(value):
            # If a value looks like json
            try:
                # convert to json
                res[key] = json.loads(value)
            except json.JSONDecodeError:
                # conversion fails, use a string
                res[key] = value
        else:
            res[key] = value
    return res


def _maybe_json(value: str) -> bool:
    """Check if a string looks like JSON."""
    stripped = value.strip()
    return stripped.startswith(("[", "{"))
