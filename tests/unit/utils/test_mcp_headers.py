"""Unit tests for MCP header parsing and utility functions."""

import json
from unittest.mock import MagicMock

from ols.constants import MCP_CLIENT_PLACEHOLDER, MCP_KUBERNETES_PLACEHOLDER
from ols.utils.mcp_headers import (
    get_servers_requiring_client_headers,
    parse_mcp_headers,
)


class TestParseMCPHeaders:
    """Tests for parse_mcp_headers function."""

    def test_parse_valid_headers_single_server(self):
        """Parse valid JSON headers for single server."""
        header_value = json.dumps({"server1": [{"Authorization": "Bearer token123"}]})
        result = parse_mcp_headers(header_value)

        assert result == {"server1": [{"Authorization": "Bearer token123"}]}

    def test_parse_valid_headers_multiple_servers(self):
        """Parse headers for multiple servers."""
        header_value = json.dumps(
            {
                "server1": [{"Authorization": "Bearer token1"}],
                "server2": [{"Authorization": "Bearer token2"}],
            }
        )
        result = parse_mcp_headers(header_value)

        assert result == {
            "server1": [{"Authorization": "Bearer token1"}],
            "server2": [{"Authorization": "Bearer token2"}],
        }

    def test_parse_server_with_multiple_header_dicts(self):
        """Parse server with multiple header dictionaries."""
        header_value = json.dumps(
            {
                "server1": [
                    {"Authorization": "Bearer token"},
                    {"X-API-Key": "key123"},
                ]
            }
        )
        result = parse_mcp_headers(header_value)

        assert result == {
            "server1": [
                {"Authorization": "Bearer token"},
                {"X-API-Key": "key123"},
            ]
        }

    def test_parse_header_dict_with_multiple_keys(self):
        """Parse header dict containing multiple key-value pairs."""
        header_value = json.dumps(
            {"server1": [{"Authorization": "Bearer token", "X-Custom": "value"}]}
        )
        result = parse_mcp_headers(header_value)

        assert result == {
            "server1": [{"Authorization": "Bearer token", "X-Custom": "value"}]
        }

    def test_parse_empty_object(self):
        """Parse empty JSON object returns empty dict."""
        header_value = json.dumps({})
        result = parse_mcp_headers(header_value)

        assert result == {}

    def test_parse_none_returns_empty(self):
        """None input returns empty dict."""
        result = parse_mcp_headers(None)

        assert result == {}

    def test_parse_empty_string_returns_empty(self):
        """Empty string returns empty dict."""
        result = parse_mcp_headers("")

        assert result == {}

    def test_parse_invalid_json_returns_empty(self):
        """Invalid JSON returns empty dict and logs warning."""
        result = parse_mcp_headers("not json")

        assert result == {}

    def test_parse_json_array_returns_empty(self):
        """JSON array instead of object returns empty dict."""
        header_value = json.dumps(["item1", "item2"])
        result = parse_mcp_headers(header_value)

        assert result == {}

    def test_parse_json_string_returns_empty(self):
        """JSON string instead of object returns empty dict."""
        header_value = json.dumps("just a string")
        result = parse_mcp_headers(header_value)

        assert result == {}

    def test_parse_server_value_not_list(self):
        """Server value that's not a list is skipped."""
        header_value = json.dumps(
            {
                "server1": {"Authorization": "Bearer token"},  # Should be list!
                "server2": [{"Authorization": "Bearer token"}],  # Correct
            }
        )
        result = parse_mcp_headers(header_value)

        # Only server2 should be included
        assert result == {"server2": [{"Authorization": "Bearer token"}]}

    def test_parse_list_item_not_dict(self):
        """List items that aren't dicts are skipped."""
        header_value = json.dumps(
            {
                "server1": [
                    {"Authorization": "Bearer token"},  # Valid
                    "not a dict",  # Invalid - should be skipped
                    {"X-Custom": "value"},  # Valid
                ]
            }
        )
        result = parse_mcp_headers(header_value)

        # Only valid dicts should be included
        assert result == {
            "server1": [
                {"Authorization": "Bearer token"},
                {"X-Custom": "value"},
            ]
        }

    def test_parse_non_string_header_keys_skipped(self):
        """Non-string keys in header dicts are skipped."""
        header_value = json.dumps(
            {
                "server1": [
                    {
                        "Authorization": "Bearer token",
                        "123": "value",
                    }  # 123 will be string in JSON
                ]
            }
        )
        result = parse_mcp_headers(header_value)

        # Both should be included since JSON converts keys to strings
        assert result == {
            "server1": [{"Authorization": "Bearer token", "123": "value"}]
        }

    def test_parse_non_string_header_values_skipped(self):
        """Non-string values in header dicts are skipped."""
        header_value = '{"server1": [{"Authorization": "Bearer token", "Count": 42}]}'
        result = parse_mcp_headers(header_value)

        # Count=42 should be skipped
        assert result == {"server1": [{"Authorization": "Bearer token"}]}

    def test_parse_nested_objects_in_values_skipped(self):
        """Nested objects in values are skipped."""
        header_value = (
            '{"server1": [{"Authorization": "Bearer token", "Nested": {"key": "val"}}]}'
        )
        result = parse_mcp_headers(header_value)

        assert result == {"server1": [{"Authorization": "Bearer token"}]}

    def test_parse_special_characters_in_values(self):
        """Special characters in header values are preserved."""
        header_value = json.dumps(
            {"server1": [{"Authorization": "Bearer token=123&foo=bar"}]}
        )
        result = parse_mcp_headers(header_value)

        assert result == {"server1": [{"Authorization": "Bearer token=123&foo=bar"}]}

    def test_parse_unicode_characters(self):
        """Unicode characters are handled correctly."""
        header_value = json.dumps({"server1": [{"X-Custom": "value-🚀"}]})
        result = parse_mcp_headers(header_value)

        assert result == {"server1": [{"X-Custom": "value-🚀"}]}

    def test_parse_whitespace_in_json(self):
        """Whitespace in JSON is handled correctly."""
        header_value = '  { "server1" : [ { "Authorization" : "Bearer token" } ] }  '
        result = parse_mcp_headers(header_value)

        assert result == {"server1": [{"Authorization": "Bearer token"}]}

    def test_parse_malformed_json_with_trailing_comma(self):
        """Malformed JSON with trailing comma returns empty dict."""
        header_value = '{"server1": [{"Authorization": "Bearer token"}],}'
        result = parse_mcp_headers(header_value)

        assert result == {}

    def test_parse_non_string_server_name_skipped(self):
        """Non-string server names are skipped (shouldn't happen in JSON)."""
        # In JSON, all keys are strings, so this is hard to test
        # But the code handles it for safety
        pass


class TestGetServersRequiringClientHeaders:
    """Tests for get_servers_requiring_client_headers function."""

    def test_no_servers_configured(self):
        """Returns empty dict when no servers configured."""
        mock_config = None
        result = get_servers_requiring_client_headers(mock_config)

        assert result == {}

    def test_empty_servers_list(self):
        """Returns empty dict when servers list is empty."""
        mock_config = MagicMock()
        mock_config.servers = []
        result = get_servers_requiring_client_headers(mock_config)

        assert result == {}

    def test_server_without_headers(self):
        """Server without headers is not included."""
        mock_server = MagicMock()
        mock_server.name = "server1"
        mock_server.headers = None

        mock_config = MagicMock()
        mock_config.servers = [mock_server]

        result = get_servers_requiring_client_headers(mock_config)

        assert result == {}

    def test_server_without_client_placeholder(self):
        """Server without client placeholder is not included."""
        mock_server = MagicMock()
        mock_server.name = "server1"
        mock_server.headers = {"Authorization": "Bearer from-file"}
        mock_server.resolved_headers = {"Authorization": "Bearer from-file"}

        mock_config = MagicMock()
        mock_config.servers = [mock_server]

        result = get_servers_requiring_client_headers(mock_config)

        assert result == {}

    def test_server_with_client_placeholder(self):
        """Server with client placeholder is included."""
        mock_server = MagicMock()
        mock_server.name = "github-mcp"
        mock_server.headers = {"Authorization": "_client_"}
        mock_server.resolved_headers = {"Authorization": MCP_CLIENT_PLACEHOLDER}

        mock_config = MagicMock()
        mock_config.servers = [mock_server]

        result = get_servers_requiring_client_headers(mock_config)

        assert result == {"github-mcp": ["Authorization"]}

    def test_server_with_multiple_client_placeholders(self):
        """Server with multiple client placeholders returns all."""
        mock_server = MagicMock()
        mock_server.name = "slack-mcp"
        mock_server.headers = {"Authorization": "_client_", "X-Slack-Team": "_client_"}
        mock_server.resolved_headers = {
            "Authorization": MCP_CLIENT_PLACEHOLDER,
            "X-Slack-Team": MCP_CLIENT_PLACEHOLDER,
        }

        mock_config = MagicMock()
        mock_config.servers = [mock_server]

        result = get_servers_requiring_client_headers(mock_config)

        assert result == {"slack-mcp": ["Authorization", "X-Slack-Team"]}

    def test_server_with_mixed_placeholders(self):
        """Server with mix of client and kubernetes placeholders."""
        mock_server = MagicMock()
        mock_server.name = "mixed-mcp"
        mock_server.headers = {
            "Authorization": "kubernetes",
            "X-API-Key": "_client_",
        }
        mock_server.resolved_headers = {
            "Authorization": MCP_KUBERNETES_PLACEHOLDER,
            "X-API-Key": MCP_CLIENT_PLACEHOLDER,
        }

        mock_config = MagicMock()
        mock_config.servers = [mock_server]

        result = get_servers_requiring_client_headers(mock_config)

        # Only client placeholder should be included
        assert result == {"mixed-mcp": ["X-API-Key"]}

    def test_multiple_servers_some_requiring_client_headers(self):
        """Multiple servers, only some require client headers."""
        mock_server1 = MagicMock()
        mock_server1.name = "server1"
        mock_server1.headers = {"Authorization": "kubernetes"}
        mock_server1.resolved_headers = {"Authorization": MCP_KUBERNETES_PLACEHOLDER}

        mock_server2 = MagicMock()
        mock_server2.name = "server2"
        mock_server2.headers = {"Authorization": "_client_"}
        mock_server2.resolved_headers = {"Authorization": MCP_CLIENT_PLACEHOLDER}

        mock_server3 = MagicMock()
        mock_server3.name = "server3"
        mock_server3.headers = None

        mock_config = MagicMock()
        mock_config.servers = [mock_server1, mock_server2, mock_server3]

        result = get_servers_requiring_client_headers(mock_config)

        # Only server2 should be included
        assert result == {"server2": ["Authorization"]}

    def test_all_servers_requiring_client_headers(self):
        """All servers require client headers."""
        mock_server1 = MagicMock()
        mock_server1.name = "server1"
        mock_server1.headers = {"Authorization": "_client_"}
        mock_server1.resolved_headers = {"Authorization": MCP_CLIENT_PLACEHOLDER}

        mock_server2 = MagicMock()
        mock_server2.name = "server2"
        mock_server2.headers = {"X-API-Key": "_client_"}
        mock_server2.resolved_headers = {"X-API-Key": MCP_CLIENT_PLACEHOLDER}

        mock_config = MagicMock()
        mock_config.servers = [mock_server1, mock_server2]

        result = get_servers_requiring_client_headers(mock_config)

        assert result == {
            "server1": ["Authorization"],
            "server2": ["X-API-Key"],
        }
