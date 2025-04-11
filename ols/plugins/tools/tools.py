"""Functions/Tools definition."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional

from langchain_core.tools.base import BaseTool

from ols.app.models.config import Tool

logger = logging.getLogger(__name__)


def check_tool_description_length(tool: BaseTool) -> None:
    """Check if the description of a tool is too long."""
    if len(tool.description) >= 1024:
        raise ValueError(f"Tool {tool.name} description too long")


@dataclass
class ToolsContext:
    """Context usable in tools."""

    user_token: Optional[str] = None


class AbstractToolSetProvider(ABC):
    """Abstract base class for tool providers."""

    @property
    @abstractmethod
    def tools(self) -> dict[str, BaseTool]:
        """Get all tools from this provider."""

    @abstractmethod
    def execute_tool(
        self, tool_name: str, args: dict, context: ToolsContext
    ) -> tuple[str, str]:
        """Execute a specific tool with the given arguments and context."""


class ToolSetProvider(AbstractToolSetProvider):
    """Tool provider."""


# TODO: configurable to also support LLMProvidersRegistry
class ToolProvidersRegistry:
    """Registry for tool providers."""

    _all_tool_providers: ClassVar = {}

    def __init__(self, enabled_tools: list[Tool]):
        """Initialize the registry with enabled tools."""
        self.enabled_tools = enabled_tools
        self.tool_providers = self.get_enabled_tool_providers(enabled_tools)
        self.all_tools = self.get_tools()
        self.tool_to_provider_mapping = self.get_tools_to_provider_mapping()

    @classmethod
    def register(cls, provider_id: str, tool_provider: Callable) -> None:
        """Register tool provider."""
        if issubclass(tool_provider, ToolSetProvider):
            cls._all_tool_providers[provider_id] = tool_provider()
            logger.debug("Tool provider '%s' registered", provider_id)
        else:
            raise TypeError(f"Unknown tool provider class: '{type(tool_provider)}'")

    def get_enabled_tool_providers(
        self, enabled_tools: list[Tool]
    ) -> list[ToolSetProvider]:
        """Get enabled tool providers."""
        return [
            self._all_tool_providers[tool_provider.name]
            for tool_provider in enabled_tools
        ]

    def get_tools(self) -> list[BaseTool]:
        """Get tools mapping from the config."""
        tools: list[BaseTool] = []
        for tool_provider in self.tool_providers:
            tools.extend(tool_provider.tools.values())
        return tools

    def get_tools_to_provider_mapping(self) -> dict[str, ToolSetProvider]:
        """Get tools to provider mapping."""
        tool_to_provider_mapping = {}
        for provider in self.tool_providers:
            for tool in provider.tools.keys():
                tool_to_provider_mapping[tool] = provider
        return tool_to_provider_mapping


def register_tool_provider_as(provider_id: str) -> Callable:
    """Register LLM provider in the `LLMProvidersRegistry`.

    Example:
    ```python
    @register_tool_provider_as("openshift")
    class OpenshiftTools(ToolProvider):
       pass
    ```
    """

    def decorator(cls: Callable) -> Callable:
        # check the tools description lenght before registering
        for tool in cls().tools.values():
            check_tool_description_length(tool)

        ToolProvidersRegistry.register(provider_id, cls)
        return cls

    return decorator
