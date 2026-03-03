"""Integration tests for streaming tool approvals (PR2)."""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel

from ols import config
from ols.app.models.config import ApprovalType
from ols.constants import MEDIA_TYPE_TEXT
from tests.mock_classes.mock_llm_loader import mock_llm_loader
from tests.mock_classes.mock_tools import mock_tools_map


class _FakeSchema(BaseModel):
    """Input schema for fake integration tools."""


class _FakeTool(StructuredTool):
    """Simple fake tool with optional metadata."""

    def __init__(self, name: str, metadata: dict | None = None) -> None:
        async def _coro(**kwargs):
            message = kwargs.get("message", "")
            return f"Tool executed successfully with args: {{'message': '{message}'}}"

        super().__init__(
            name=name,
            description=f"Fake tool {name}",
            func=lambda **kwargs: f"Tool executed successfully with args: {kwargs}",
            coroutine=_coro,
            args_schema=_FakeSchema,
            metadata=metadata,
        )


@pytest.fixture(scope="function")
def _setup() -> None:
    """Set up test client for approval integration tests."""
    config.reload_from_yaml_file("tests/config/config_for_integration_tests.yaml")
    from ols.app.main import app  # pylint: disable=C0415

    pytest.client = TestClient(app)


def _single_tool_side_effect_factory():
    """Create side effect that emits one tool-call round then final answer."""
    call_count = 0

    async def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            yield AIMessageChunk(
                content="",
                response_metadata={"finish_reason": "tool_calls"},
                tool_call_chunks=[
                    {"name": "get_namespaces_mock", "args": "{}", "id": "call_a"}
                ],
                tool_calls=[
                    {"name": "get_namespaces_mock", "args": {}, "id": "call_a"}
                ],
            )
            return
        if call_count == 2:
            yield AIMessageChunk(content="done")
            return
        yield AIMessageChunk(content="", response_metadata={"finish_reason": "stop"})

    return _side_effect


def _three_tools_side_effect_factory():
    """Create side effect that emits three tool calls then final answer."""
    call_count = 0

    async def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            yield AIMessageChunk(
                content="",
                response_metadata={"finish_reason": "tool_calls"},
                tool_call_chunks=[
                    {
                        "name": "mock_tool_client",
                        "args": '{"message": "Hello World"}',
                        "id": "call_client",
                    },
                    {
                        "name": "mock_tool_file",
                        "args": '{"message": "Hello World"}',
                        "id": "call_file",
                    },
                    {
                        "name": "mock_tool_k8s",
                        "args": '{"message": "Hello World"}',
                        "id": "call_k8s",
                    },
                ],
                tool_calls=[
                    {
                        "name": "mock_tool_client",
                        "args": {"message": "Hello World"},
                        "id": "call_client",
                    },
                    {
                        "name": "mock_tool_file",
                        "args": {"message": "Hello World"},
                        "id": "call_file",
                    },
                    {
                        "name": "mock_tool_k8s",
                        "args": {"message": "Hello World"},
                        "id": "call_k8s",
                    },
                ],
            )
            return
        if call_count == 2:
            yield AIMessageChunk(content="done")
            return
        yield AIMessageChunk(content="", response_metadata={"finish_reason": "stop"})

    return _side_effect


def test_streaming_approval_always_emits_approval_and_rejection(_setup) -> None:
    """Verify approval_type=always emits approval_required and rejected tool result."""
    with (
        patch("ols.app.endpoints.ols.validate_question", return_value=True),
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new=AsyncMock(return_value=mock_tools_map),
        ),
        patch(
            "ols.src.query_helpers.query_helper.load_llm",
            new=mock_llm_loader(None),
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm",
            side_effect=_single_tool_side_effect_factory(),
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.TokenHandler.calculate_and_check_available_tokens",
            return_value=1000,
        ),
        patch(
            "ols.src.tools.tools.config.tools_approval.approval_type",
            ApprovalType.ALWAYS,
        ),
        patch("ols.src.tools.tools.config.tools_approval.approval_timeout", 1),
        patch(
            "ols.src.tools.tools.get_approval_decision",
            new=AsyncMock(return_value="rejected"),
        ),
    ):
        response = pytest.client.post(
            "/v1/streaming_query",
            json={"query": "run a tool", "media_type": MEDIA_TYPE_TEXT},
        )

    assert response.status_code == 200
    assert "Approval request:" in response.text
    assert "execution was rejected" in response.text


def test_streaming_approval_never_skips_approval_and_executes(_setup) -> None:
    """Verify approval_type=never does not emit approval_required events."""
    with (
        patch("ols.app.endpoints.ols.validate_question", return_value=True),
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new=AsyncMock(return_value=mock_tools_map),
        ),
        patch(
            "ols.src.query_helpers.query_helper.load_llm",
            new=mock_llm_loader(None),
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm",
            side_effect=_single_tool_side_effect_factory(),
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.TokenHandler.calculate_and_check_available_tokens",
            return_value=1000,
        ),
        patch(
            "ols.src.tools.tools.config.tools_approval.approval_type",
            ApprovalType.NEVER,
        ),
        patch(
            "ols.src.tools.tools.get_approval_decision",
            new=AsyncMock(
                side_effect=AssertionError("approval wait should not be called")
            ),
        ),
    ):
        response = pytest.client.post(
            "/v1/streaming_query",
            json={"query": "run a tool", "media_type": MEDIA_TYPE_TEXT},
        )

    assert response.status_code == 200
    assert "Approval request:" not in response.text
    assert "execution was rejected" not in response.text
    assert "Tool result:" in response.text


def test_streaming_approval_tool_annotations_mixed_tools(_setup) -> None:
    """Verify tool_annotations strategy approves only non-read-only tools."""
    mixed_tools = [
        _FakeTool(
            "mock_tool_client",
            metadata={"annotations": {"readOnlyHint": False, "otherHint": "client"}},
        ),
        _FakeTool("mock_tool_file", metadata={"annotations": {"readOnlyHint": True}}),
        _FakeTool("mock_tool_k8s"),
    ]

    with (
        patch("ols.app.endpoints.ols.validate_question", return_value=True),
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new=AsyncMock(return_value=mixed_tools),
        ),
        patch(
            "ols.src.query_helpers.query_helper.load_llm",
            new=mock_llm_loader(None),
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm",
            side_effect=_three_tools_side_effect_factory(),
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.TokenHandler.calculate_and_check_available_tokens",
            return_value=1000,
        ),
        patch(
            "ols.src.tools.tools.config.tools_approval.approval_type",
            ApprovalType.TOOL_ANNOTATIONS,
        ),
        patch("ols.src.tools.tools.config.tools_approval.approval_timeout", 1),
        patch(
            "ols.src.tools.tools.get_approval_decision",
            new=AsyncMock(return_value="rejected"),
        ),
    ):
        response = pytest.client.post(
            "/v1/streaming_query",
            json={"query": "run three tools", "media_type": MEDIA_TYPE_TEXT},
        )

    assert response.status_code == 200
    assert response.text.count("Approval request:") == 2
    assert '"tool_name": "mock_tool_client"' in response.text
    assert '"tool_name": "mock_tool_k8s"' in response.text
    assert "Tool 'mock_tool_file' execution was rejected" not in response.text
