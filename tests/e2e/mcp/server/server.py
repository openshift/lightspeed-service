#!/usr/bin/env python3
"""Minimal MCP mock server for testing authorization headers.

This is a simple HTTP server that implements basic MCP protocol endpoints
for testing purposes. It captures and logs authorization headers, making it
useful for validating that Lightspeed Core Stack correctly sends auth headers
to MCP servers.

Usage:
    python server.py [port]

Example:
    python server.py 3000
"""

import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

# Global storage for captured headers (last request)
last_headers: dict[str, str] = {}
request_log: list = []


class MCPMockHandler(BaseHTTPRequestHandler):
    """HTTP request handler for mock MCP server."""

    def log_message(
        self, format: str, *args: Any  # noqa: A002  # pylint: disable=redefined-builtin
    ) -> None:
        """Log requests with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {format % args}")

    def _capture_headers(self) -> None:
        """Capture all headers from the request."""
        last_headers.clear()

        # Capture all headers for debugging
        last_headers.update(self.headers.items())

        # Log the request
        request_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "method": self.command,
                "path": self.path,
                "headers": dict(last_headers),
            }
        )

        # Keep only last 10 requests
        if len(request_log) > 10:
            request_log.pop(0)

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Handle POST requests (MCP protocol endpoints)."""
        self._capture_headers()

        # Read request body to get JSON-RPC request
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            request_data = json.loads(body.decode("utf-8"))
            request_id = request_data.get("id", 1)
            method = request_data.get("method", "unknown")
        except (json.JSONDecodeError, UnicodeDecodeError):
            request_id = 1
            method = "unknown"

        # Determine tool name based on authorization header to avoid collisions
        auth_header = self.headers.get("Authorization", "")

        tool_name: str | None = None
        tool_desc = ""
        tool_annotations: dict[str, Any] | None = None

        match auth_header:
            case _ if "test-secret-token" in auth_header:
                tool_name = "openshift_cluster_status"
                tool_desc = "Check OpenShift cluster health and status"
                tool_annotations = {"readOnlyHint": True}
            case _ if (
                "my-client-token" in auth_header
                or "streaming-client-token" in auth_header
            ):
                tool_name = "openshift_route_info"
                tool_desc = "Get route details for an OpenShift application"
                tool_annotations = {"readOnlyHint": False, "otherHint": "client"}
            case _ if auth_header:
                tool_name = "openshift_pod_logs"
                tool_desc = "Retrieve pod logs from an OpenShift namespace"

        # Handle MCP protocol methods
        if method == "initialize":
            # Return MCP initialize response
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                    },
                    "serverInfo": {
                        "name": "mock-mcp-server",
                        "version": "1.0.0",
                    },
                },
            }
        elif method == "tools/list":
            tools: list[dict[str, Any]] = []
            if tool_name is not None:
                tool_definition: dict[str, Any] = {
                    "name": tool_name,
                    "description": tool_desc,
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "OpenShift resource name or query",
                            }
                        },
                    },
                }
                if tool_annotations is not None:
                    tool_definition["annotations"] = tool_annotations
                tools.append(tool_definition)

            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": tools},
            }
        elif method == "tools/call":
            # Handle tool execution
            # Extract tool arguments from request
            tool_args = request_data.get("params", {}).get("arguments", {})
            # Return simple string result (langchain-mcp-adapters will extract from content)
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Tool executed successfully with args: {tool_args}",
                        }
                    ],
                    "isError": False,
                },
            }
        else:
            # Generic success response for other methods
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"status": "ok"},
            }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

        print(f"  → Captured headers: {last_headers}")

    def do_GET(self) -> None:  # pylint: disable=invalid-name
        """Handle GET requests (debug endpoints)."""
        match self.path:
            case "/debug/headers":
                self._send_json_response(
                    {"last_headers": last_headers, "request_count": len(request_log)}
                )
            case "/debug/requests":
                self._send_json_response(request_log)
            case "/":
                self._send_help_page()
            case _:
                self.send_response(404)
                self.end_headers()

    def do_DELETE(self) -> None:  # pylint: disable=invalid-name
        """Handle DELETE requests (clear debug state)."""
        if self.path == "/debug/requests":
            request_log.clear()
            last_headers.clear()
            self._send_json_response({"status": "cleared"})
        else:
            self.send_response(404)
            self.end_headers()

    def _send_json_response(self, data: dict | list) -> None:
        """Send a JSON response."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def _send_help_page(self) -> None:
        """Send HTML help page for root endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        help_html = """<!DOCTYPE html>
        <html>
        <head><title>MCP Mock Server</title></head>
        <body>
            <h1>MCP Mock Server</h1>
            <p>Development mock server for testing MCP integrations.</p>
            <h2>Debug Endpoints:</h2>
            <ul>
                <li><a href="/debug/headers">/debug/headers</a> - View captured headers</li>
                <li><a href="/debug/requests">/debug/requests</a> - View request log</li>
            </ul>
            <h2>MCP Protocol:</h2>
            <p>POST requests to any path with JSON-RPC format:</p>
            <ul>
                <li><code>{"jsonrpc": "2.0", "id": 1, "method": "initialize"}</code></li>
                <li><code>{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}</code></li>
            </ul>
        </body>
        </html>
        """
        self.wfile.write(help_html.encode())


def main() -> None:
    """Start the mock MCP server."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

    server = HTTPServer(("", port), MCPMockHandler)

    print("=" * 70)
    print(f"MCP Mock Server listening on http://localhost:{port}")
    print("=" * 70)
    print("Debug endpoints:")
    print("  • /debug/headers  - View captured headers")
    print("  • /debug/requests - View request log")
    print("MCP endpoint:")
    print("  • POST to any path (e.g., / or /mcp/v1/list_tools)")
    print("=" * 70)
    print("Press Ctrl+C to stop")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down mock server...")
        server.shutdown()


if __name__ == "__main__":
    main()
