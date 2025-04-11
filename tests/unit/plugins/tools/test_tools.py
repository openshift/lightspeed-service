"""Unit tests for tools module."""

import pytest
from langchain.tools import tool

from ols.app.models.config import Tool
from ols.plugins.tools.tools import (
    AbstractToolSetProvider,
    ToolProvidersRegistry,
    ToolsContext,
    ToolSetProvider,
    check_tool_description_length,
    register_tool_provider_as,
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


def test_tools_context():
    """Test ToolsContext."""
    context = ToolsContext(user_token="fake-token")  # noqa: S106
    assert context.user_token == "fake-token"  # noqa: S105

    # test default value
    context = ToolsContext()
    assert context.user_token is None


def test_abstract_tool_set_provider():
    """Test AbstractToolSetProvider."""

    class BadToolSetProvider(AbstractToolSetProvider):
        pass

    with pytest.raises(TypeError):
        BadToolSetProvider()

    class GoodToolSetProvider(AbstractToolSetProvider):
        """Concrete implementation of AbstractToolSetProvider."""

        def tools(self):
            return []

        def execute_tool(self, tool_name, args, context):
            return "result"

    provider = GoodToolSetProvider()
    assert provider.tools() == []
    assert provider.execute_tool("tool_name", ["arg1"], ToolsContext()) == "result"


def test_valid_tool_provider_is_registered():
    """Test valid (`ToolSetProvider` subclass) is registered."""

    @register_tool_provider_as("spam")
    class Spam(ToolSetProvider):
        @property
        def tools(self):
            return {}

        def execute_tool(self, *args, **kwargs):
            pass

    assert "spam" in ToolProvidersRegistry._all_tool_providers


def test_invalid_tool_provider_is_not_registered():
    """Test raise when invalid (not `LLMProvider` subclass) is registered."""
    with pytest.raises(
        TypeError, match="Unknown tool provider class: '<class 'type'>'"
    ):

        @register_tool_provider_as("spam")
        class Spam:
            @property
            def tools(self):
                return {}

            def execute_tool(self, *args, **kwargs):
                pass


def test_tool_providers_registry():
    """Test ToolProvidersRegistry."""

    @tool
    def fake_tool_1():
        """Fake tool description."""
        pass

    @tool
    def fake_tool_2():
        """Fake tool description."""
        pass

    @register_tool_provider_as("fake1")
    class FakeToolProvider1(ToolSetProvider):
        @property
        def tools(self):
            return {"tool1": fake_tool_1}

        def execute_tool(self, *args, **kwargs):
            return "result"

    @register_tool_provider_as("fake2")
    class FakeToolProvider2(ToolSetProvider):
        @property
        def tools(self):
            return {"tool2": fake_tool_2}

        def execute_tool(self, *args, **kwargs):
            return "result"

    registry_with_no_allowed_tools = ToolProvidersRegistry(enabled_tools=[])
    assert registry_with_no_allowed_tools.tool_providers == []
    assert registry_with_no_allowed_tools.all_tools == []
    assert registry_with_no_allowed_tools.tool_to_provider_mapping == {}

    registry_with_tools = ToolProvidersRegistry(
        enabled_tools=[
            Tool(name="fake1", type="tool-set"),
            Tool(name="fake2", type="tool-set"),
        ]
    )
    assert len(registry_with_tools.tool_providers) == 2
    assert isinstance(registry_with_tools.tool_providers[0], FakeToolProvider1)
    assert isinstance(registry_with_tools.tool_providers[1], FakeToolProvider2)
    assert registry_with_tools.all_tools == [fake_tool_1, fake_tool_2]
    assert len(registry_with_tools.tool_to_provider_mapping) == 2
    assert isinstance(
        registry_with_tools.tool_to_provider_mapping["tool1"], FakeToolProvider1
    )
    assert isinstance(
        registry_with_tools.tool_to_provider_mapping["tool2"], FakeToolProvider2
    )
