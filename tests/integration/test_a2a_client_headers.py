"""Integration tests for A2A client headers feature.

These tests verify the A2A client headers functionality by:
1. Configuring OLS with different A2A auth types (file-based, client-provided)
2. Testing both /v1/query and /v1/streaming_query endpoints
3. Verifying correct header resolution and agent selection

The A2A card resolver is mocked to avoid actual network connections, but we test
that the correct agents are contacted based on header availability.

Test Coverage:
- Query without client headers (graceful degradation - skips client-auth agent)
- Query WITH client headers (includes client-auth agent)
- Streaming query without client headers
- Streaming query WITH client headers
"""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

from unittest.mock import patch

import pytest
from a2a.types import AgentCard, AgentSkill
from fastapi.testclient import TestClient

from ols import config
from tests.mock_classes.mock_langchain_interface import mock_langchain_interface
from tests.mock_classes.mock_llm_loader import mock_llm_loader

A2A_FETCH_CARD_PATH = "ols.src.a2a.client._fetch_agent_card"


def _make_card(agent_name: str) -> AgentCard:
    """Build a minimal AgentCard for testing."""
    return AgentCard(
        name=agent_name,
        description=f"Test agent {agent_name}",
        url=f"http://{agent_name}:8080",
        version="1.0",
        skills=[
            AgentSkill(
                id="test-skill",
                name="Test Skill",
                description="A skill for integration testing",
                tags=[],
            )
        ],
        capabilities={},
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )


@pytest.fixture(scope="function")
def _setup() -> None:
    """Set up the test client for A2A integration tests."""
    config.reload_from_yaml_file("tests/config/config_for_a2a_integration_tests.yaml")
    config.k8s_a2a_agents_resolved = False

    from ols.app.main import app  # pylint: disable=import-outside-toplevel

    pytest.client = TestClient(app)  # type: ignore[attr-defined]


def test_query_without_client_headers(_setup: None) -> None:
    """Test query without client headers - mock-client-auth agent should be skipped."""
    ml = mock_langchain_interface(None)

    async def _fake_fetch_card(
        base_url: str, headers: dict, request_timeout: int
    ) -> AgentCard:
        """Return a card only for file-auth agent."""
        if "localhost:14000" in base_url:
            return _make_card("mock-file-auth")
        raise AssertionError(f"Unexpected agent URL: {base_url}")

    with (
        patch("ols.src.query_helpers.query_helper.load_llm", new=mock_llm_loader(ml)),
        patch(A2A_FETCH_CARD_PATH, side_effect=_fake_fetch_card) as mock_fetch,
    ):
        response = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/query",
            json={"query": "What tools are available?"},
        )

        assert response.status_code == 200

        call_urls = [call.args[0] for call in mock_fetch.call_args_list]
        assert any("localhost:14000" in url for url in call_urls)
        assert not any("localhost:14001" in url for url in call_urls)


def test_query_with_client_headers(_setup: None) -> None:
    """Test query with client headers - both agents should be contacted."""
    ml = mock_langchain_interface(None)

    async def _fake_fetch_card(
        base_url: str, headers: dict, request_timeout: int
    ) -> AgentCard:
        """Return cards for both agents."""
        if "localhost:14000" in base_url:
            return _make_card("mock-file-auth")
        if "localhost:14001" in base_url:
            return _make_card("mock-client-auth")
        raise AssertionError(f"Unexpected agent URL: {base_url}")

    with (
        patch("ols.src.query_helpers.query_helper.load_llm", new=mock_llm_loader(ml)),
        patch(A2A_FETCH_CARD_PATH, side_effect=_fake_fetch_card) as mock_fetch,
    ):
        mcp_headers = {
            "mock-client-auth": {"Authorization": "Bearer my-client-token-456"}
        }

        response = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/query",
            json={"query": "What tools are available?", "mcp_headers": mcp_headers},
        )

        assert response.status_code == 200

        call_urls = [call.args[0] for call in mock_fetch.call_args_list]
        assert any("localhost:14000" in url for url in call_urls)
        assert any("localhost:14001" in url for url in call_urls)


def test_query_client_headers_resolve_correctly(_setup: None) -> None:
    """Test that client-provided headers are passed through to the agent card fetch."""
    ml = mock_langchain_interface(None)

    captured_headers: dict[str, dict] = {}

    async def _fake_fetch_card(
        base_url: str, headers: dict, request_timeout: int
    ) -> AgentCard:
        """Capture headers for verification."""
        if "localhost:14000" in base_url:
            captured_headers["mock-file-auth"] = headers
            return _make_card("mock-file-auth")
        if "localhost:14001" in base_url:
            captured_headers["mock-client-auth"] = headers
            return _make_card("mock-client-auth")
        raise AssertionError(f"Unexpected agent URL: {base_url}")

    with (
        patch("ols.src.query_helpers.query_helper.load_llm", new=mock_llm_loader(ml)),
        patch(A2A_FETCH_CARD_PATH, side_effect=_fake_fetch_card),
    ):
        mcp_headers = {
            "mock-client-auth": {"Authorization": "Bearer my-client-token-456"}
        }

        response = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/query",
            json={"query": "What tools are available?", "mcp_headers": mcp_headers},
        )

        assert response.status_code == 200

        assert captured_headers["mock-file-auth"]["Authorization"] == (
            "Bearer test-file-token-123"
        )
        assert captured_headers["mock-client-auth"]["Authorization"] == (
            "Bearer my-client-token-456"
        )


def test_streaming_query_without_client_headers(_setup: None) -> None:
    """Test streaming query without client headers - client-auth agent should be skipped."""
    ml = mock_langchain_interface(None)

    async def _fake_fetch_card(
        base_url: str, headers: dict, request_timeout: int
    ) -> AgentCard:
        """Return a card only for file-auth agent."""
        if "localhost:14000" in base_url:
            return _make_card("mock-file-auth")
        raise AssertionError(f"Unexpected agent URL: {base_url}")

    with (
        patch("ols.src.query_helpers.query_helper.load_llm", new=mock_llm_loader(ml)),
        patch(A2A_FETCH_CARD_PATH, side_effect=_fake_fetch_card) as mock_fetch,
    ):
        response = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/streaming_query",
            json={"query": "What tools are available?"},
        )

        assert response.status_code == 200

        call_urls = [call.args[0] for call in mock_fetch.call_args_list]
        assert any("localhost:14000" in url for url in call_urls)
        assert not any("localhost:14001" in url for url in call_urls)


def test_streaming_query_with_client_headers(_setup: None) -> None:
    """Test streaming query with client headers - both agents should be contacted."""
    ml = mock_langchain_interface(None)

    async def _fake_fetch_card(
        base_url: str, headers: dict, request_timeout: int
    ) -> AgentCard:
        """Return cards for both agents."""
        if "localhost:14000" in base_url:
            return _make_card("mock-file-auth")
        if "localhost:14001" in base_url:
            return _make_card("mock-client-auth")
        raise AssertionError(f"Unexpected agent URL: {base_url}")

    with (
        patch("ols.src.query_helpers.query_helper.load_llm", new=mock_llm_loader(ml)),
        patch(A2A_FETCH_CARD_PATH, side_effect=_fake_fetch_card) as mock_fetch,
    ):
        mcp_headers = {
            "mock-client-auth": {"Authorization": "Bearer streaming-client-token-789"}
        }

        response = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/streaming_query",
            json={"query": "What tools are available?", "mcp_headers": mcp_headers},
        )

        assert response.status_code == 200

        call_urls = [call.args[0] for call in mock_fetch.call_args_list]
        assert any("localhost:14000" in url for url in call_urls)
        assert any("localhost:14001" in url for url in call_urls)
