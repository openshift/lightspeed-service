"""Unit tests for A2A client: tool discovery, wrapping, and AgentsRAG."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import (
    AgentCard,
    AgentSkill,
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel, Field

from ols.src.a2a.client import (
    AgentsRAG,
    _build_agent_tool,
    _extract_message_text,
    _extract_task_text,
    _sanitize_tool_name,
    get_a2a_tools,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_skill():
    """Create a sample A2A AgentSkill."""
    return AgentSkill(
        id="troubleshoot",
        name="Troubleshoot",
        description="Diagnose OpenShift cluster issues",
        tags=[],
    )


@pytest.fixture
def sample_card(sample_skill):
    """Create a sample A2A AgentCard with one skill."""
    return AgentCard(
        name="OLS Remote",
        description="Remote OLS agent for testing",
        url="http://remote-agent:8080",
        version="1.0",
        skills=[sample_skill],
        capabilities={},
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )


@pytest.fixture
def mock_agent_cfg():
    """Create a mock A2A agent config."""
    cfg = MagicMock()
    cfg.name = "remote-agent"
    cfg.url = "http://remote-agent:8080"
    cfg.timeout = 15
    cfg.headers = {}
    cfg.resolved_headers = {}
    return cfg


@pytest.fixture
def mock_tool():
    """Create a mock StructuredTool with A2A metadata."""

    class ToolInput(BaseModel):
        query: str = Field(description="query")

    return StructuredTool(
        name="remote_agent_troubleshoot",
        description="Diagnose OpenShift cluster issues",
        func=lambda query: "",
        args_schema=ToolInput,
        metadata={"a2a_agent": "remote-agent", "a2a_skill_id": "troubleshoot"},
    )


# ---------------------------------------------------------------------------
# _sanitize_tool_name
# ---------------------------------------------------------------------------


class TestSanitizeToolName:
    """Tests for _sanitize_tool_name."""

    def test_alphanumeric_passes_through(self):
        """Verify alphanumeric and underscore names are unchanged."""
        assert _sanitize_tool_name("hello_world") == "hello_world"

    def test_special_chars_replaced(self):
        """Verify hyphens, dots, and slashes are replaced with underscores."""
        assert _sanitize_tool_name("my-agent.skill/v2") == "my_agent_skill_v2"

    def test_spaces_replaced(self):
        """Verify spaces are replaced with underscores."""
        assert _sanitize_tool_name("agent name") == "agent_name"


# ---------------------------------------------------------------------------
# _extract_task_text / _extract_message_text
# ---------------------------------------------------------------------------


class TestExtractText:
    """Tests for text extraction helpers."""

    def test_extract_task_text_from_artifacts(self):
        """Verify artifact text is extracted from a completed task."""
        task = Task(
            id="t1",
            contextId="ctx1",
            status=TaskStatus(state=TaskState.completed),
            artifacts=[
                Artifact(
                    artifactId="a1",
                    parts=[Part(root=TextPart(text="artifact output"))],
                )
            ],
        )
        assert _extract_task_text(task) == "artifact output"

    def test_extract_task_text_falls_back_to_status_message(self):
        """Verify status message is used when no artifacts are present."""
        task = Task(
            id="t1",
            contextId="ctx1",
            status=TaskStatus(
                state=TaskState.completed,
                message=Message(
                    message_id="m1",
                    role=Role.agent,
                    parts=[Part(root=TextPart(text="status fallback"))],
                ),
            ),
        )
        assert _extract_task_text(task) == "status fallback"

    def test_extract_task_text_empty_when_no_text(self):
        """Verify empty string is returned when task has no text content."""
        task = Task(
            id="t1",
            contextId="ctx1",
            status=TaskStatus(state=TaskState.completed),
        )
        assert _extract_task_text(task) == ""

    def test_extract_message_text(self):
        """Verify text parts of a message are joined with newlines."""
        msg = Message(
            message_id="m1",
            role=Role.agent,
            parts=[
                Part(root=TextPart(text="line1")),
                Part(root=TextPart(text="line2")),
            ],
        )
        assert _extract_message_text(msg) == "line1\nline2"


# ---------------------------------------------------------------------------
# _build_agent_tool
# ---------------------------------------------------------------------------


class TestBuildAgentTool:
    """Tests for _build_agent_tool."""

    def test_returns_structured_tool(self, sample_skill, sample_card):
        """Verify _build_agent_tool returns a StructuredTool with correct metadata."""
        tool = _build_agent_tool("remote-agent", sample_skill, sample_card, {}, 30)
        assert isinstance(tool, StructuredTool)
        assert tool.name == "remote_agent_troubleshoot"
        assert tool.description == sample_skill.description
        assert tool.metadata["a2a_agent"] == "remote-agent"
        assert tool.metadata["a2a_skill_id"] == "troubleshoot"

    @pytest.mark.asyncio
    async def test_invoke_returns_task_text(self, sample_skill, sample_card):
        """Verify invoking the tool returns extracted task text."""
        tool = _build_agent_tool("remote-agent", sample_skill, sample_card, {}, 30)

        task = Task(
            id="t1",
            contextId="ctx1",
            status=TaskStatus(state=TaskState.completed),
            artifacts=[
                Artifact(
                    artifactId="a1",
                    parts=[Part(root=TextPart(text="tool result"))],
                )
            ],
        )

        mock_client = MagicMock()
        mock_client.send_message.return_value = AsyncIterFromList([(task, None)])
        mock_client.close = AsyncMock()

        with patch("ols.src.a2a.client.ClientFactory") as mock_factory_cls:
            mock_factory_cls.return_value.create.return_value = mock_client
            result = await tool.ainvoke({"query": "why is pod crashing?"})

        assert result == "tool result"
        mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invoke_raises_on_failed_task(self, sample_skill, sample_card):
        """Verify RuntimeError is raised when the remote agent fails."""
        tool = _build_agent_tool("remote-agent", sample_skill, sample_card, {}, 30)

        task = Task(
            id="t1",
            contextId="ctx1",
            status=TaskStatus(state=TaskState.failed),
        )

        mock_client = MagicMock()
        mock_client.send_message.return_value = AsyncIterFromList([(task, None)])
        mock_client.close = AsyncMock()

        with (
            patch("ols.src.a2a.client.ClientFactory") as mock_factory_cls,
            pytest.raises(RuntimeError, match="Remote agent returned an error"),
        ):
            mock_factory_cls.return_value.create.return_value = mock_client
            await tool.ainvoke({"query": "fail"})


# ---------------------------------------------------------------------------
# _gather_a2a_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGatherA2ATools:
    """Tests for _gather_a2a_tools."""

    async def test_discovers_tools_from_agent(self, mock_agent_cfg, sample_card):
        """Verify tools are discovered from an agent's card skills."""
        with (
            patch("ols.src.a2a.client.resolve_server_headers", return_value={}),
            patch(
                "ols.src.a2a.client._fetch_agent_card",
                new_callable=AsyncMock,
                return_value=sample_card,
            ),
        ):
            from ols.src.a2a.client import _gather_a2a_tools

            agents_config, tools = await _gather_a2a_tools([mock_agent_cfg])

        assert "remote-agent" in agents_config
        assert len(tools) == 1
        assert tools[0].name == "remote_agent_troubleshoot"

    async def test_skips_agent_when_headers_unresolvable(self, mock_agent_cfg):
        """Verify agent is skipped when headers cannot be resolved."""
        with patch("ols.src.a2a.client.resolve_server_headers", return_value=None):
            from ols.src.a2a.client import _gather_a2a_tools

            agents_config, tools = await _gather_a2a_tools([mock_agent_cfg])

        assert agents_config == {}
        assert tools == []

    async def test_skips_agent_on_fetch_error(self, mock_agent_cfg):
        """Verify agent is skipped when card fetch raises an exception."""
        with (
            patch("ols.src.a2a.client.resolve_server_headers", return_value={}),
            patch(
                "ols.src.a2a.client._fetch_agent_card",
                new_callable=AsyncMock,
                side_effect=Exception("connection refused"),
            ),
        ):
            from ols.src.a2a.client import _gather_a2a_tools

            agents_config, tools = await _gather_a2a_tools([mock_agent_cfg])

        assert agents_config == {}
        assert tools == []

    async def test_populates_rag_when_requested(self, mock_agent_cfg, sample_card):
        """Verify AgentsRAG.populate_agents is called when populate_to_rag=True."""
        mock_rag = MagicMock()
        with (
            patch("ols.src.a2a.client.resolve_server_headers", return_value={}),
            patch(
                "ols.src.a2a.client._fetch_agent_card",
                new_callable=AsyncMock,
                return_value=sample_card,
            ),
            patch("ols.src.a2a.client.config") as mock_config,
        ):
            mock_config.agents_rag = mock_rag
            from ols.src.a2a.client import _gather_a2a_tools

            await _gather_a2a_tools([mock_agent_cfg], populate_to_rag=True)

        mock_rag.populate_agents.assert_called_once()


# ---------------------------------------------------------------------------
# get_a2a_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetA2ATools:
    """Tests for get_a2a_tools."""

    async def test_returns_empty_when_no_agents_configured(self):
        """Verify empty list when no A2A agents are configured."""
        with patch("ols.src.a2a.client.config") as mock_config:
            mock_config.a2a_agents.agents = []
            result = await get_a2a_tools("test query")
        assert result == []

    async def test_without_agents_rag(self, mock_agent_cfg, mock_tool):
        """Verify all tools are returned when agents_rag is disabled."""
        with (
            patch("ols.src.a2a.client.config") as mock_config,
            patch(
                "ols.src.a2a.client._gather_a2a_tools",
                new_callable=AsyncMock,
                return_value=({"remote-agent": mock_agent_cfg}, [mock_tool]),
            ),
        ):
            mock_config.a2a_agents.agents = [mock_agent_cfg]
            mock_config.agents_rag = None

            result = await get_a2a_tools("test query")

        assert len(result) == 1
        assert result[0].name == "remote_agent_troubleshoot"

    async def test_with_agents_rag_first_call(self, mock_agent_cfg, mock_tool):
        """Verify RAG is populated on the first call and tools are filtered."""
        with (
            patch("ols.src.a2a.client.config") as mock_config,
            patch(
                "ols.src.a2a.client._gather_a2a_tools",
                new_callable=AsyncMock,
                return_value=({"remote-agent": mock_agent_cfg}, [mock_tool]),
            ),
        ):
            mock_config.a2a_agents.agents = [mock_agent_cfg]
            mock_config.agents_rag = MagicMock()
            mock_config.agents_rag.populate_agents = MagicMock()
            mock_config.agents_rag.set_default_agents = MagicMock()
            mock_config.agents_rag.retrieve_hybrid.return_value = {
                "remote-agent": [{"name": "remote_agent_troubleshoot", "desc": "test"}]
            }
            mock_config.k8s_a2a_agents_resolved = False
            mock_config.a2a_agents_dict = {"remote-agent": mock_agent_cfg}

            result = await get_a2a_tools(
                "test query", user_token="k8s-token"  # noqa: S106
            )

        assert len(result) == 1

    async def test_rag_filtering_error_fallback(self, mock_agent_cfg, mock_tool):
        """Verify all tools are returned when RAG filtering raises an error."""
        with (
            patch("ols.src.a2a.client.config") as mock_config,
            patch(
                "ols.src.a2a.client._gather_a2a_tools",
                new_callable=AsyncMock,
                return_value=({"remote-agent": mock_agent_cfg}, [mock_tool]),
            ),
        ):
            mock_config.a2a_agents.agents = [mock_agent_cfg]
            mock_config.agents_rag = MagicMock()
            mock_config.agents_rag.retrieve_hybrid.side_effect = Exception("RAG error")
            mock_config.k8s_a2a_agents_resolved = True

            result = await get_a2a_tools("test query")

        assert len(result) == 1
        assert result[0].name == "remote_agent_troubleshoot"

    async def test_no_matching_tools(self, mock_agent_cfg):
        """Verify empty list when RAG returns no matching tools."""
        with patch("ols.src.a2a.client.config") as mock_config:
            mock_config.a2a_agents.agents = [mock_agent_cfg]
            mock_config.agents_rag = MagicMock()
            mock_config.agents_rag.retrieve_hybrid.return_value = {}
            mock_config.k8s_a2a_agents_resolved = True

            result = await get_a2a_tools("test query")

        assert result == []

    async def test_with_client_headers(self, mock_agent_cfg, mock_tool):
        """Verify client_headers agents are included in RAG retrieval."""
        with (
            patch("ols.src.a2a.client.config") as mock_config,
            patch(
                "ols.src.a2a.client._gather_a2a_tools",
                new_callable=AsyncMock,
                return_value=({"remote-agent": mock_agent_cfg}, [mock_tool]),
            ),
        ):
            mock_config.a2a_agents.agents = [mock_agent_cfg]
            mock_config.agents_rag = MagicMock()
            mock_config.agents_rag.retrieve_hybrid.return_value = {
                "remote-agent": [{"name": "remote_agent_troubleshoot", "desc": "test"}]
            }
            mock_config.k8s_a2a_agents_resolved = True
            mock_config.a2a_agents_dict = {"remote-agent": mock_agent_cfg}

            client_headers = {"remote-agent": {"Authorization": "Bearer client-tok"}}
            result = await get_a2a_tools(
                "test query",
                user_token="k8s-token",  # noqa: S106
                client_headers=client_headers,
            )

        mock_config.agents_rag.retrieve_hybrid.assert_called_once()
        call_kwargs = mock_config.agents_rag.retrieve_hybrid.call_args[1]
        assert call_kwargs["client_agents"] == ["remote-agent"]
        assert len(result) == 1


# ---------------------------------------------------------------------------
# AgentsRAG
# ---------------------------------------------------------------------------


class TestAgentsRAG:
    """Tests for the AgentsRAG class."""

    @pytest.fixture
    def agents_rag(self):
        """Create an AgentsRAG with a placeholder encoder."""
        return AgentsRAG(
            encode_fn=lambda text: [0.1] * 8,
            alpha=0.8,
            top_k=10,
            threshold=0.0,
        )

    def test_set_default_agents(self, agents_rag):
        """Verify default allowed agents are stored as a set."""
        agents_rag.set_default_agents(["agent-a", "agent-b"])
        assert agents_rag.default_allowed_agents == {"agent-a", "agent-b"}

    def test_populate_and_retrieve(self, agents_rag, mock_tool):
        """Verify tools can be populated into RAG and retrieved by query."""
        agents_rag.set_default_agents(["remote-agent"])
        agents_rag.populate_agents([mock_tool])

        result = agents_rag.retrieve_hybrid("troubleshoot cluster")
        assert "remote-agent" in result
        tool_dicts = result["remote-agent"]
        assert len(tool_dicts) >= 1
        assert tool_dicts[0]["name"] == "remote_agent_troubleshoot"

    def test_convert_agent_to_dict(self, agents_rag, mock_tool):
        """Verify tool is converted to dict with name, server, and desc."""
        d = agents_rag._convert_agent_to_dict(mock_tool)
        assert d["name"] == "remote_agent_troubleshoot"
        assert d["server"] == "remote-agent"
        assert d["desc"] == "Diagnose OpenShift cluster issues"

    def test_build_text(self, agents_rag):
        """Verify _build_text concatenates name and description."""
        t = {"name": "my_tool", "desc": "does things"}
        assert agents_rag._build_text(t) == "my_tool does things"


# ---------------------------------------------------------------------------
# Async iterator helper for mocking client.send_message
# ---------------------------------------------------------------------------


class AsyncIterFromList:
    """Wrap a list as an async iterator for mocking."""

    def __init__(self, items: list) -> None:
        """Initialize with a list of items to iterate."""
        self._items = items
        self._idx = 0

    def __aiter__(self):  # noqa: D105
        return self

    async def __anext__(self):  # noqa: D105
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item
