"""Integration tests for MCP client headers feature.

These tests verify the MCP client headers functionality by:
1. Configuring OLS with different MCP auth types (file-based, client-provided)
2. Testing both /v1/query and /v1/streaming_query endpoints
3. Verifying correct header resolution and server selection

The MCP client is mocked to avoid actual network connections, but we test
that the correct servers are selected based on header availability.

Test Coverage:
- Discovery endpoint (/v1/mcp/client-auth-headers)
- Query without client headers (graceful degradation - skips client-auth server)
- Query WITH client headers (includes client-auth server)
- Streaming query without client headers
- Streaming query WITH client headers
"""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ols import config
from tests.mock_classes.mock_langchain_interface import mock_langchain_interface
from tests.mock_classes.mock_llm_loader import mock_llm_loader

# Patch path for MultiServerMCPClient - it's imported in ols.utils.mcp_utils
MCP_CLIENT_PATCH_PATH = "ols.utils.mcp_utils.MultiServerMCPClient"


@pytest.fixture(scope="function")
def _setup() -> None:
    """Set up the test client for MCP integration tests."""
    config.reload_from_yaml_file("tests/config/config_for_mcp_integration_tests.yaml")

    from ols.app.main import app  # pylint: disable=import-outside-toplevel

    pytest.client = TestClient(app)  # type: ignore[attr-defined]


def test_mcp_discovery_endpoint(_setup: None) -> None:
    """Test the /v1/mcp/client-auth-headers discovery endpoint."""
    response = pytest.client.get("/v1/mcp/client-auth-headers")  # type: ignore[attr-defined]

    assert response.status_code == 200
    data = response.json()

    assert "servers" in data
    assert len(data["servers"]) == 1
    assert data["servers"][0]["server_name"] == "mock-client-auth"
    assert data["servers"][0]["required_headers"] == ["Authorization"]


def test_query_without_client_headers(_setup: None) -> None:
    """Test query without client headers - mock-client-auth should be skipped."""
    ml = mock_langchain_interface(None)

    # Mock the MCP client to avoid actual network connections

    mock_mcp_client_instance = MagicMock()
    mock_mcp_client_instance.list_tools = AsyncMock(return_value=[])
    mock_mcp_client_instance.close = AsyncMock()

    with (
        patch("ols.src.query_helpers.query_helper.load_llm", new=mock_llm_loader(ml)),
        patch(
            MCP_CLIENT_PATCH_PATH,
            return_value=mock_mcp_client_instance,
        ) as mock_mcp_client_class,
    ):
        response = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/query",
            json={"query": "What tools are available?"},
        )

        # Request should succeed
        assert response.status_code == 200

        # Verify MCP client was called with only file-auth server (no client-auth)
        mock_mcp_client_class.assert_called_once()
        call_args = mock_mcp_client_class.call_args
        mcp_config = call_args[0][0]

        # Should only have mock-file-auth (client-auth requires client headers)
        assert "mock-file-auth" in mcp_config
        assert "mock-client-auth" not in mcp_config
        assert mcp_config["mock-file-auth"]["headers"]["Authorization"] == (
            "Bearer test-file-token-123"
        )


def test_query_with_client_headers(_setup: None) -> None:
    """Test query with client headers - both servers should be contacted."""
    ml = mock_langchain_interface(None)

    # Mock the MCP client

    mock_mcp_client_instance = MagicMock()
    mock_mcp_client_instance.list_tools = AsyncMock(return_value=[])
    mock_mcp_client_instance.close = AsyncMock()

    with (
        patch("ols.src.query_helpers.query_helper.load_llm", new=mock_llm_loader(ml)),
        patch(
            MCP_CLIENT_PATCH_PATH,
            return_value=mock_mcp_client_instance,
        ) as mock_mcp_client_class,
    ):
        mcp_headers = {
            "mock-client-auth": {"Authorization": "Bearer my-client-token-456"}
        }

        response = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/query",
            json={"query": "What tools are available?", "mcp_headers": mcp_headers},
        )

        # Request should succeed with client headers
        assert response.status_code == 200

        # Verify MCP client was initialized with correct config including headers
        mock_mcp_client_class.assert_called_once()
        call_args = mock_mcp_client_class.call_args
        mcp_config = call_args[0][0]

        # Verify both servers are included
        assert "mock-file-auth" in mcp_config
        assert "mock-client-auth" in mcp_config

        # Verify the file-auth header
        assert mcp_config["mock-file-auth"]["headers"]["Authorization"] == (
            "Bearer test-file-token-123"
        )

        # Verify the client-provided header was passed through
        assert "headers" in mcp_config["mock-client-auth"]
        assert (
            mcp_config["mock-client-auth"]["headers"]["Authorization"]
            == "Bearer my-client-token-456"
        )


def test_streaming_query_without_client_headers(_setup: None) -> None:
    """Test streaming query without client headers."""
    ml = mock_langchain_interface(None)

    # Mock the MCP client

    mock_mcp_client_instance = MagicMock()
    mock_mcp_client_instance.list_tools = AsyncMock(return_value=[])
    mock_mcp_client_instance.close = AsyncMock()

    with (
        patch("ols.src.query_helpers.query_helper.load_llm", new=mock_llm_loader(ml)),
        patch(
            MCP_CLIENT_PATCH_PATH,
            return_value=mock_mcp_client_instance,
        ) as mock_mcp_client_class,
    ):
        response = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/streaming_query",
            json={"query": "What tools are available?"},
        )

        # Request should succeed
        assert response.status_code == 200

        # Verify MCP client was called with only file-auth server
        mock_mcp_client_class.assert_called_once()
        call_args = mock_mcp_client_class.call_args
        mcp_config = call_args[0][0]

        # Should only have mock-file-auth (client-auth requires client headers)
        assert "mock-file-auth" in mcp_config
        assert "mock-client-auth" not in mcp_config


def test_streaming_query_with_client_headers(_setup: None) -> None:
    """Test streaming query with client headers."""
    ml = mock_langchain_interface(None)

    # Mock the MCP client

    mock_mcp_client_instance = MagicMock()
    mock_mcp_client_instance.list_tools = AsyncMock(return_value=[])
    mock_mcp_client_instance.close = AsyncMock()

    with (
        patch("ols.src.query_helpers.query_helper.load_llm", new=mock_llm_loader(ml)),
        patch(
            MCP_CLIENT_PATCH_PATH,
            return_value=mock_mcp_client_instance,
        ) as mock_mcp_client_class,
    ):
        mcp_headers = {
            "mock-client-auth": {"Authorization": "Bearer streaming-client-token-789"}
        }

        response = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/streaming_query",
            json={"query": "What tools are available?", "mcp_headers": mcp_headers},
        )

        # Request should succeed with client headers
        assert response.status_code == 200

        # Verify MCP client was initialized with correct config including headers
        mock_mcp_client_class.assert_called_once()
        call_args = mock_mcp_client_class.call_args
        mcp_config = call_args[0][0]

        # Verify both servers are included
        assert "mock-file-auth" in mcp_config
        assert "mock-client-auth" in mcp_config

        # Verify the file-auth header
        assert mcp_config["mock-file-auth"]["headers"]["Authorization"] == (
            "Bearer test-file-token-123"
        )

        # Verify the client-provided header was passed through
        assert "headers" in mcp_config["mock-client-auth"]
        assert (
            mcp_config["mock-client-auth"]["headers"]["Authorization"]
            == "Bearer streaming-client-token-789"
        )
