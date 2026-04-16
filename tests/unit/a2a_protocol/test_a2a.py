"""Unit tests for A2A protocol support."""

from unittest.mock import MagicMock, patch

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)
from fastapi import FastAPI

from ols import config
from ols.constants import QueryMode

config.ols_config.authentication_config.module = "k8s"

from ols.app.routers import _mount_a2a_routes  # noqa: E402
from ols.src.a2a.server import (  # noqa: E402
    SKILL_ASK,
    SKILL_ID_ASK,
    SKILL_ID_TO_MODE,
    SKILL_ID_TROUBLESHOOTING,
    SKILL_TROUBLESHOOTING,
    OLSAgentExecutor,
    build_agent_card,
)

# --- Agent Card tests ---


def test_build_agent_card_has_required_fields():
    """Verify the agent card contains all A2A-required fields."""
    card = build_agent_card("https://ols.example.com")

    assert card.name == "OpenShift LightSpeed"
    assert card.url == "https://ols.example.com"
    assert card.version
    assert card.description
    assert card.capabilities is not None
    assert card.default_input_modes == ["text/plain"]
    assert card.default_output_modes == ["text/plain"]


def test_build_agent_card_has_two_skills():
    """Verify the agent card advertises ask and troubleshooting skills."""
    card = build_agent_card("https://ols.example.com")

    assert len(card.skills) == 2
    skill_ids = {s.id for s in card.skills}
    assert skill_ids == {SKILL_ID_ASK, SKILL_ID_TROUBLESHOOTING}


def test_build_agent_card_capabilities():
    """Verify streaming is enabled and push notifications are disabled."""
    card = build_agent_card("https://ols.example.com")

    assert card.capabilities.streaming is True
    assert card.capabilities.push_notifications is False


def test_build_agent_card_provider():
    """Verify the provider is Red Hat."""
    card = build_agent_card("https://ols.example.com")

    assert card.provider.organization == "Red Hat"
    assert card.provider.url == "https://www.redhat.com"


def test_build_agent_card_uses_provided_url():
    """Verify the card URL matches the provided server_url."""
    card = build_agent_card("https://custom.host:9443")
    assert card.url == "https://custom.host:9443"


def test_ask_skill_is_valid():
    """Verify the ask skill constant is well-formed."""
    assert SKILL_ASK.id == SKILL_ID_ASK
    assert SKILL_ASK.name
    assert SKILL_ASK.description
    assert len(SKILL_ASK.tags) >= 2
    assert len(SKILL_ASK.examples) >= 2


def test_troubleshooting_skill_is_valid():
    """Verify the troubleshooting skill constant is well-formed."""
    assert SKILL_TROUBLESHOOTING.id == SKILL_ID_TROUBLESHOOTING
    assert SKILL_TROUBLESHOOTING.name
    assert SKILL_TROUBLESHOOTING.description
    assert "troubleshooting" in SKILL_TROUBLESHOOTING.tags


def test_skill_id_to_mode_mapping():
    """Verify skill IDs map to the correct QueryMode values."""
    assert SKILL_ID_TO_MODE[SKILL_ID_ASK] == QueryMode.ASK
    assert SKILL_ID_TO_MODE[SKILL_ID_TROUBLESHOOTING] == QueryMode.TROUBLESHOOTING


# --- Executor tests ---


def _make_context(text: str, metadata: dict | None = None) -> RequestContext:
    """Create a RequestContext with a user message containing the given text."""
    message = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=text))],
        message_id="msg-1",
    )
    params = MessageSendParams(message=message, metadata=metadata)
    return RequestContext(request=params)


def _make_event_queue() -> EventQueue:
    """Create an EventQueue for capturing events."""
    return EventQueue()


@pytest.mark.asyncio
async def test_execute_empty_query_fails_task():
    """Verify that an empty query results in a failed task status."""
    executor = OLSAgentExecutor()
    context = _make_context("   ")
    queue = _make_event_queue()

    await executor.execute(context, queue)

    event = await queue.dequeue_event()
    assert isinstance(event, TaskStatusUpdateEvent)
    assert event.status.state == TaskState.failed


@pytest.mark.asyncio
@patch("ols.src.a2a.server.DocsSummarizer")
@patch("ols.src.a2a.server.config")
async def test_execute_success_default_mode(mock_config, mock_summarizer_class):
    """Verify a query without skill_id defaults to ask mode."""
    from ols.app.models.models import StreamChunkType, StreamedChunk

    async def fake_generate_response(*args, **kwargs):
        yield StreamedChunk(type=StreamChunkType.TEXT, text="Hello from OLS")
        yield StreamedChunk(type=StreamChunkType.END, data={})

    mock_instance = MagicMock()
    mock_instance.generate_response = fake_generate_response
    mock_summarizer_class.return_value = mock_instance
    mock_config.rag_index_loader.get_retriever.return_value = None

    executor = OLSAgentExecutor()
    context = _make_context("How do I create a deployment?")
    queue = _make_event_queue()

    await executor.execute(context, queue)

    mock_summarizer_class.assert_called_once_with(mode=QueryMode.ASK, streaming=True)

    event1 = await queue.dequeue_event()
    assert isinstance(event1, TaskStatusUpdateEvent)
    assert event1.status.state == TaskState.working

    event2 = await queue.dequeue_event()
    assert isinstance(event2, TaskArtifactUpdateEvent)
    assert event2.artifact.parts[0].root.text == "Hello from OLS"

    event3 = await queue.dequeue_event()
    assert isinstance(event3, TaskStatusUpdateEvent)
    assert event3.status.state == TaskState.completed


@pytest.mark.asyncio
@patch("ols.src.a2a.server.DocsSummarizer")
@patch("ols.src.a2a.server.config")
async def test_execute_troubleshooting_mode(mock_config, mock_summarizer_class):
    """Verify skill_id in metadata selects troubleshooting mode."""
    from ols.app.models.models import StreamChunkType, StreamedChunk

    async def fake_generate_response(*args, **kwargs):
        yield StreamedChunk(type=StreamChunkType.TEXT, text="Check pod events")
        yield StreamedChunk(type=StreamChunkType.END, data={})

    mock_instance = MagicMock()
    mock_instance.generate_response = fake_generate_response
    mock_summarizer_class.return_value = mock_instance
    mock_config.rag_index_loader.get_retriever.return_value = None

    executor = OLSAgentExecutor()
    context = _make_context(
        "Why is my pod in CrashLoopBackOff?",
        metadata={"skill_id": SKILL_ID_TROUBLESHOOTING},
    )
    queue = _make_event_queue()

    await executor.execute(context, queue)

    mock_summarizer_class.assert_called_once_with(
        mode=QueryMode.TROUBLESHOOTING, streaming=True
    )

    event1 = await queue.dequeue_event()
    assert isinstance(event1, TaskStatusUpdateEvent)
    assert event1.status.state == TaskState.working

    event2 = await queue.dequeue_event()
    assert isinstance(event2, TaskArtifactUpdateEvent)
    assert event2.artifact.parts[0].root.text == "Check pod events"


@pytest.mark.asyncio
@patch("ols.src.a2a.server.DocsSummarizer")
@patch("ols.src.a2a.server.config")
async def test_execute_summarizer_error_fails_task(mock_config, mock_summarizer_class):
    """Verify that a summarizer exception results in a failed task."""

    async def failing_generator(*args, **kwargs):
        raise RuntimeError("LLM error")
        yield

    mock_instance = MagicMock()
    mock_instance.generate_response = failing_generator
    mock_summarizer_class.return_value = mock_instance
    mock_config.rag_index_loader.get_retriever.return_value = None

    executor = OLSAgentExecutor()
    context = _make_context("Some query")
    queue = _make_event_queue()

    await executor.execute(context, queue)

    event1 = await queue.dequeue_event()
    assert isinstance(event1, TaskStatusUpdateEvent)
    assert event1.status.state == TaskState.working

    event2 = await queue.dequeue_event()
    assert isinstance(event2, TaskStatusUpdateEvent)
    assert event2.status.state == TaskState.failed


@pytest.mark.asyncio
async def test_cancel():
    """Verify cancel publishes a canceled status event."""
    executor = OLSAgentExecutor()
    context = _make_context("anything")
    queue = _make_event_queue()

    await executor.cancel(context, queue)

    event = await queue.dequeue_event()
    assert isinstance(event, TaskStatusUpdateEvent)
    assert event.status.state == TaskState.canceled


# --- Route mounting tests ---


@patch("a2a.server.apps.A2AFastAPIApplication")
@patch("a2a.server.request_handlers.DefaultRequestHandler")
@patch("a2a.server.events.InMemoryQueueManager")
@patch("a2a.server.tasks.InMemoryTaskStore")
@patch("ols.src.a2a.server.build_agent_card")
@patch("ols.src.a2a.server.OLSAgentExecutor")
def test_mount_a2a_routes_adds_routes(
    mock_executor_class,
    mock_build_card,
    mock_task_store,
    mock_queue_manager,
    mock_handler_class,
    mock_a2a_app_class,
):
    """Verify _mount_a2a_routes integrates A2A SDK with the FastAPI app."""
    app = FastAPI()
    mock_a2a_instance = MagicMock()
    mock_a2a_app_class.return_value = mock_a2a_instance

    _mount_a2a_routes(app)

    mock_build_card.assert_called_once()
    mock_executor_class.assert_called_once()
    mock_handler_class.assert_called_once()
    mock_a2a_app_class.assert_called_once()
    mock_a2a_instance.add_routes_to_app.assert_called_once_with(
        app,
        agent_card_url="/.well-known/agent-card.json",
        rpc_url="/a2a",
    )
