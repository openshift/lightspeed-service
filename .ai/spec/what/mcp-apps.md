# MCP Apps

MCP Apps are a UI integration layer that allows the OpenShift console plugin
to discover renderable resources and invoke tool calls on MCP servers directly,
bypassing the LLM query pipeline. This is distinct from tool execution during
LLM conversations (see `what/tools.md`).

## Behavioral Rules

### Resource Fetching

1. The system must expose a `POST /v1/mcp-apps/resources` endpoint that
   proxies resource reads to a named MCP server on behalf of the console UI.

2. Only resources with the `ui://` URI scheme are valid. The system must
   reject any resource URI that does not start with `ui://` or is empty
   after the scheme prefix, returning HTTP 400.

3. A resource request must specify the `server_name` identifying which
   configured MCP server owns the resource, and the `resource_uri` to fetch.

4. The system must open a Streamable HTTP MCP session to the target server,
   call `read_resource` with the requested URI, and return the first content
   block from the response.

5. Resource content must be classified as either `"text"` (for text-based
   content such as HTML, JS, CSS) or `"blob"` (for binary content such as
   base64-encoded data). The MIME type must default to `text/html` when the
   MCP server does not provide one.

6. Resource-level metadata from the MCP server (CSP directives, permissions,
   or other hints) must be forwarded to the client in the response `meta`
   field.

7. If the MCP server returns an empty content list for the requested URI,
   the system must return HTTP 404.

### Direct Tool Calls

8. The system must expose a `POST /v1/mcp-apps/tools/call` endpoint that
   proxies tool calls to a named MCP server outside the LLM conversation
   flow.

9. A tool call request must specify the `server_name`, `tool_name`, and
   `arguments` (defaulting to an empty dict).

10. The system must open a Streamable HTTP MCP session to the target server,
    call the specified tool with the provided arguments, and return the
    result.

11. Tool call results must be returned as a list of typed content blocks.
    Supported content block types are `text`, `image`, and `audio`.
    Unsupported content types must be logged and omitted from the response.

12. Structured content returned by the MCP server (e.g., rich data for UI
    rendering) must be preserved and forwarded to the client separately from
    the content block list.

13. The `is_error` flag from the MCP tool result must be forwarded to the
    client so the UI can distinguish successful calls from error responses.

### Client Auth Header Discovery

14. The system must expose a `GET /v1/mcp/client-auth-headers` endpoint that
    tells clients which MCP servers require client-provided authentication
    headers.

15. The response must list each server that has at least one header configured
    with the `"client"` placeholder, along with the names of the required
    headers.

16. Clients use this information to include the correct headers in subsequent
    resource fetch or tool call requests via the `mcp_headers` field.

### Authentication and Header Resolution

17. All MCP Apps endpoints require the same authentication as the main query
    endpoint (kubernetes token review via the `/ols-access` virtual path).

18. Header resolution for MCP Apps follows the same rules as for LLM tool
    calls (see `what/tools.md`, rules 2--3): `"kubernetes"` placeholders
    resolve to the user's bearer token, `"client"` placeholders resolve from
    the request's `mcp_headers` field, and literal values pass through.

19. If header resolution fails for the target server (missing kubernetes token
    or missing required client header), the system must return HTTP 401 with a
    message identifying the server.

20. If no MCP servers are configured, or the requested server name does not
    match any configured server, the system must return HTTP 404.

### Connection Parameters

21. Each MCP Apps connection must use the target server's configured URL and
    per-server timeout. When no timeout is configured for a server, the
    default is 30 seconds.

22. Any unhandled exception during resource fetching or tool execution must
    return HTTP 500 with the error detail. The error must be logged.

## Configuration Surface

MCP Apps share MCP server configuration with the tool execution pipeline.
See `what/tools.md` for the full `mcp_servers.servers[]` configuration table.

| Field Path | Type | Default | Purpose |
|---|---|---|---|
| `mcp_servers.servers[].name` | string | required | Server identifier used in `server_name` request field |
| `mcp_servers.servers[].url` | string | required | Server HTTP endpoint for Streamable HTTP sessions |
| `mcp_servers.servers[].timeout` | int | 30 (apps) | Per-server timeout in seconds for app requests |
| `mcp_servers.servers[].headers` | map | {} | Authorization headers; `"client"` placeholders drive discovery |

## Constraints

1. **MCP Apps are not LLM tool calls.** Resource fetches and direct tool calls
   bypass the LLM query pipeline entirely. They do not participate in
   conversation history, tool approval workflows, token budgeting, or retry
   policies. Those behaviors apply only to tool calls within the LLM
   generation loop (see `what/tools.md`).

2. **Only `ui://` URIs are accepted.** The resource endpoint must never proxy
   arbitrary URI schemes. This is enforced by validation before any MCP
   session is opened.

3. **Server must be configured.** Both resource and tool call endpoints
   require the target server to be present in `mcp_servers.servers[]`.
   There is no server auto-discovery; the admin must explicitly configure
   every MCP server.

4. **One content block per resource.** The resource endpoint returns only the
   first content block from the MCP server response. Multi-block resources
   are not supported.

5. **No retry or fault isolation across servers.** Unlike tool gathering in
   the query pipeline, MCP Apps target a single named server per request.
   If that server is unavailable, the request fails. There is no retry
   logic or fallback.

6. **Shared auth boundary.** MCP Apps use the same authentication gate as the
   query endpoint. A user who cannot access `/ols-access` cannot use MCP
   Apps.

## Planned Changes

| Jira Key | Summary |
|---|---|
| OLS-2657 | MCP-apps UI productization strategy -- define the supported surface area, security model, and lifecycle for MCP-powered console apps |
| OLS-2684 | Remove client MCP headers -- eliminating the `"client"` header placeholder will affect the client-auth-headers discovery endpoint and `mcp_headers` field (shared with `what/tools.md`) |
