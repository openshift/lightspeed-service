# REST API

The REST API is the only external interface to the OpenShift LightSpeed service. Every capability -- queries, conversations, feedback, tools, health checks, and metrics -- is exposed through HTTP endpoints defined here.

## Behavioral Rules

### General

1. All paths are relative to the service root (e.g. `https://host:port`).
2. All endpoints under `/v1` (except `/v1/feedback/status`) require authentication via bearer token in the `Authorization` header. The token is resolved into a user identity consisting of a user ID (UUID), a username, a `skip_user_id_check` flag, and a user token.
3. All authenticated endpoints (except `/metrics`) enforce the `ols-access` authorization scope. The `/metrics` endpoint enforces a separate `ols-metrics-access` scope.
4. Health probes (`/readiness`, `/liveness`) require no authentication and have no version prefix.
5. Extra fields in request bodies are rejected with HTTP 422 for endpoints that use `"extra": "forbid"` in their Pydantic model (`LLMRequest`, `MCPAppResourceRequest`, `MCPAppToolCallRequest`, `ToolApprovalDecisionRequest`).
6. The `conversation_id` field, wherever it appears, must be a valid SUID (UUID format). Invalid format returns HTTP 400.

### Query Endpoints

7. `POST /v1/query` accepts a question and returns a complete JSON response after the LLM finishes generating. [PLANNED: OLS-2682] This endpoint will be removed; streaming will become the only query path.
8. `POST /v1/streaming_query` accepts the same request body as `/v1/query` and returns a streaming response using Server-Sent Events (SSE). The response `Content-Type` matches the request's `media_type` field.
9. Both query endpoints share the same request processing pipeline: authenticate, retrieve/generate conversation ID, redact query and attachments, validate provider/model, check quota, append attachments, then invoke the LLM. See `what/query-processing.md` for pipeline behavior details.
10. Both query endpoints store conversation history and (if enabled) transcripts after the response is generated.
11. Token consumption is recorded against all configured quota limiters after each query. The response includes remaining quota per limiter.

### Conversation Endpoints

12. `GET /v1/conversations` returns all conversations belonging to the authenticated user.
13. `GET /v1/conversations/{conversation_id}` returns the full chat history for one conversation, including messages, tool calls, and tool results per exchange.
14. `DELETE /v1/conversations/{conversation_id}` deletes a conversation. A request to delete a non-existent conversation returns 200 with `success: false`, not 404.
15. `PUT /v1/conversations/{conversation_id}` updates the topic summary of an existing conversation. The conversation must exist (404 if not).

### Feedback Endpoints

16. `GET /v1/feedback/status` returns whether feedback collection is enabled. No authentication required.
17. `POST /v1/feedback` stores user feedback. Requires authentication and feedback collection to be enabled (403 if disabled).

### MCP Endpoints

18. `POST /v1/mcp-apps/resources` fetches a `ui://` resource from a configured MCP server. The URI must start with `ui://` and contain a path after the prefix.
19. `POST /v1/mcp-apps/tools/call` proxies a tool call to a configured MCP server.
20. `GET /v1/mcp/client-auth-headers` returns which MCP servers require client-provided authorization headers so clients know what to include in the `mcp_headers` field of query requests.
21. `POST /v1/tool-approvals/decision` submits an approval or rejection for a pending tool execution request. Used during streaming queries when a tool call requires explicit user approval.

### Infrastructure Endpoints

22. `POST /authorized` validates the caller's credentials and authorization. The authentication check itself is the purpose of this endpoint. No `/v1` prefix.
23. `GET /readiness` checks three subsystems: RAG index loaded (if configured), default LLM reachable, and conversation cache ready. All three must pass. The LLM readiness result is cached for a configurable duration. Returns 503 with cause if any subsystem fails.
24. `GET /liveness` always returns `alive: true` if the process is running.
25. `GET /metrics` returns Prometheus metrics in exposition format. Requires `ols-metrics-access` scope. No version prefix.

---

## Endpoint Reference

### POST /v1/query

Submit a question and receive a complete JSON response.

**Auth**: Required (`ols-access` scope)

#### Request body (LLMRequest)

| Field           | Type                                         | Required | Default        | Description |
|-----------------|----------------------------------------------|----------|----------------|-------------|
| query           | string                                       | yes      | --             | The user's question |
| conversation_id | string (UUID)                                | no       | auto-generated | Existing conversation to continue |
| provider        | string                                       | no       | server default | LLM provider name |
| model           | string                                       | no       | server default | LLM model name |
| system_prompt   | string                                       | no       | null           | Override the default system prompt |
| attachments     | array of Attachment                          | no       | null           | Supplementary content (see Attachments section) |
| media_type      | string                                       | no       | `"text/plain"` | Must be `text/plain` or `application/json` |
| mcp_headers     | object (server name -> header key/value map) | no       | null           | Client-provided auth headers for MCP servers |
| mode            | string                                       | no       | `"ask"`        | Query mode: `ask` or `troubleshooting` |

**Validation rules:**
- `provider` and `model` must both be specified or both omitted. Specifying only one returns 422.
- If specified, the provider/model pair must match a configured LLM provider (422 if not).
- `media_type` must be exactly `text/plain` or `application/json` (422 if not).
- Each attachment's `attachment_type` and `content_type` must be in the allowed sets (422 if not).
- The caller's token quota is checked before processing. Exhausted quota returns 500.
- The query and all attachment content are redacted (PII removal) before processing.

[PLANNED: OLS-2682] This endpoint will be removed in favor of streaming-only.

#### Response 200 (LLMResponse)

| Field                | Type                        | Description |
|----------------------|-----------------------------|-------------|
| conversation_id      | string (UUID)               | The conversation this exchange belongs to |
| response             | string                      | The LLM's answer |
| referenced_documents | array of ReferencedDocument | Documents used to generate the answer (deduplicated, order-preserving) |
| truncated            | boolean                     | True if conversation history was truncated to fit the context window |
| input_tokens         | integer                     | Tokens sent to the LLM |
| output_tokens        | integer                     | Tokens received from the LLM |
| available_quotas     | object (limiter name -> int)| Remaining quota per configured limiter |
| tool_calls           | array of object             | Tool invocations made during response generation |
| tool_results         | array of object             | Results from tool invocations |

**ReferencedDocument:**

| Field     | Type   | Description |
|-----------|--------|-------------|
| doc_url   | string | URL of the documentation page |
| doc_title | string | Title of the documentation page |

#### Error responses

| Status | Condition |
|--------|-----------|
| 400    | Invalid conversation ID format |
| 401    | Missing or invalid credentials |
| 403    | Caller lacks permission |
| 413    | Prompt exceeds the LLM context window limit |
| 422    | Invalid provider/model pair, invalid attachment type/content type, invalid media_type, or extra fields |
| 500    | LLM unreachable, quota database error, quota exceeded, query redaction failure, conversation storage failure, or other internal error |

---

### POST /v1/streaming_query

Submit a question and receive the response as a server-sent event stream. Accepts the same request body as `/v1/query`.

**Auth**: Required (`ols-access` scope)

#### Request body

Identical to `POST /v1/query` (LLMRequest).

#### Response 200

A streaming response whose `Content-Type` matches the request's `media_type` field. See the Streaming Format section for event types and wire formats.

#### Error responses

| Status | Condition |
|--------|-----------|
| 401    | Missing or invalid credentials |
| 403    | Caller lacks permission |
| 500    | Internal error during request setup |

Errors that occur after the stream has started (prompt too long, LLM failure) are delivered as in-stream error events rather than HTTP status codes.

---

### GET /v1/conversations

List all conversations belonging to the authenticated user.

**Auth**: Required (`ols-access` scope)

#### Response 200 (ConversationsListResponse)

| Field         | Type                       | Description |
|---------------|----------------------------|-------------|
| conversations | array of ConversationData  | The user's conversations |

**ConversationData:**

| Field                  | Type                     | Description |
|------------------------|--------------------------|-------------|
| conversation_id        | string (UUID)            | Unique conversation identifier |
| topic_summary          | string                   | User-assigned summary (default empty string) |
| last_message_timestamp | number (float, unix)     | Timestamp of the most recent message |
| message_count          | integer                  | Number of message exchanges (default 0) |

#### Error responses

| Status | Condition |
|--------|-----------|
| 401    | Missing or invalid credentials |
| 403    | Caller lacks permission |
| 500    | Internal error |

---

### GET /v1/conversations/{conversation_id}

Retrieve the full chat history for a specific conversation.

**Auth**: Required (`ols-access` scope)

**Path parameter**: `conversation_id` -- must be a valid UUID (SUID format).

#### Response 200 (ConversationDetailResponse)

| Field           | Type                   | Description |
|-----------------|------------------------|-------------|
| conversation_id | string (UUID)          | The conversation identifier |
| chat_history    | array of ChatExchange  | Ordered list of message exchanges |

**ChatExchange:**

| Field        | Type             | Description |
|--------------|------------------|-------------|
| messages     | array of Message | A pair of user and assistant messages |
| tool_calls   | array of object  | Tool invocations made during this exchange |
| tool_results | array of object  | Results from tool invocations |

**Message:**

| Field   | Type   | Description |
|---------|--------|-------------|
| type    | string | `"user"` or `"assistant"` |
| content | string | Message text |

#### Error responses

| Status | Condition |
|--------|-----------|
| 400    | Invalid conversation ID format |
| 401    | Missing or invalid credentials |
| 403    | Caller lacks permission |
| 404    | Conversation does not exist |
| 500    | Internal error |

---

### DELETE /v1/conversations/{conversation_id}

Delete a conversation and its history.

**Auth**: Required (`ols-access` scope)

**Path parameter**: `conversation_id` -- must be a valid UUID.

#### Response 200 (ConversationDeleteResponse)

| Field           | Type    | Description |
|-----------------|---------|-------------|
| conversation_id | string  | The conversation that was targeted |
| response        | string  | Human-readable result message |
| success         | boolean | True if the conversation existed and was deleted; false if not found |

A request to delete a non-existent conversation returns 200 with `success: false`, not 404.

#### Error responses

| Status | Condition |
|--------|-----------|
| 400    | Invalid conversation ID format |
| 401    | Missing or invalid credentials |
| 403    | Caller lacks permission |
| 500    | Internal error |

---

### PUT /v1/conversations/{conversation_id}

Update the topic summary of an existing conversation.

**Auth**: Required (`ols-access` scope)

**Path parameter**: `conversation_id` -- must be a valid UUID.

#### Request body (ConversationUpdateRequest)

| Field         | Type   | Required | Description |
|---------------|--------|----------|-------------|
| topic_summary | string | yes      | The new topic summary |

#### Response 200 (ConversationUpdateResponse)

| Field           | Type    | Description |
|-----------------|---------|-------------|
| conversation_id | string  | The conversation that was updated |
| success         | boolean | True if the update succeeded |
| message         | string  | Human-readable result message |

#### Error responses

| Status | Condition |
|--------|-----------|
| 400    | Invalid conversation ID format |
| 401    | Missing or invalid credentials |
| 403    | Caller lacks permission |
| 404    | Conversation does not exist |
| 500    | Internal error |

---

### GET /v1/feedback/status

Check whether feedback collection is enabled.

**Auth**: None required

#### Response 200 (StatusResponse)

| Field         | Type   | Description |
|---------------|--------|-------------|
| functionality | string | Always `"feedback"` |
| status        | object | `{"enabled": true}` or `{"enabled": false}` |

---

### POST /v1/feedback

Submit user feedback about a conversation exchange.

**Auth**: Required (`ols-access` scope)

**Precondition**: Feedback collection must be enabled. If disabled, returns 403.

#### Request body (FeedbackRequest)

| Field           | Type          | Required | Default | Description |
|-----------------|---------------|----------|---------|-------------|
| conversation_id | string (UUID) | yes      | --      | The conversation being rated |
| user_question   | string        | yes      | --      | The question the user asked |
| llm_response    | string        | yes      | --      | The response the LLM gave |
| sentiment       | integer       | no       | null    | Rating: must be exactly `-1` or `1` |
| user_feedback   | string        | no       | null    | Free-text feedback |

**Validation rules:**
- `conversation_id` must be a valid UUID.
- `sentiment`, if provided, must be exactly `-1` or `1`.
- At least one of `sentiment` or `user_feedback` must be provided.

#### Response 200 (FeedbackResponse)

| Field    | Type   | Description |
|----------|--------|-------------|
| response | string | Always `"feedback received"` |

#### Error responses

| Status | Condition |
|--------|-----------|
| 401    | Missing or invalid credentials |
| 403    | Caller lacks permission, or feedback is disabled |
| 422    | Validation error (invalid UUID, invalid sentiment, neither sentiment nor feedback provided) |
| 500    | Feedback storage failure |

---

### POST /v1/mcp-apps/resources

Fetch a `ui://` resource from a configured MCP server. Used by the console to load app content (HTML, JS, CSS) served by MCP servers.

**Auth**: Required (`ols-access` scope)

#### Request body (MCPAppResourceRequest)

| Field        | Type                                         | Required | Default | Description |
|--------------|----------------------------------------------|----------|---------|-------------|
| resource_uri | string                                       | yes      | --      | Must start with `ui://` and contain a path after the prefix |
| server_name  | string                                       | yes      | --      | Name of the MCP server as defined in service configuration |
| mcp_headers  | object (server name -> header key/value map) | no       | null    | Client-provided auth headers for the MCP server |

Extra fields are rejected (422).

#### Response 200 (MCPAppResourceResponse)

| Field        | Type                     | Description |
|--------------|--------------------------|-------------|
| uri          | string                   | The URI of the returned resource |
| mime_type    | string                   | MIME type of the content (e.g. `text/html`) |
| content      | string                   | The resource content (text or base64-encoded blob) |
| content_type | string (`text` or `blob`)| Whether `content` is plaintext or base64 |
| meta         | object or null           | Server-provided metadata (CSP policies, permissions, etc.) |

#### Error responses

| Status | Condition |
|--------|-----------|
| 400    | Invalid resource URI (does not start with `ui://` or path is empty) |
| 401    | Missing credentials, or required MCP server credentials not provided |
| 403    | Caller lacks permission |
| 404    | MCP server not configured, or resource not found on the server |
| 500    | Communication failure with the MCP server |

---

### POST /v1/mcp-apps/tools/call

Proxy a tool call to a configured MCP server. Used by app iframes to invoke MCP tools.

**Auth**: Required (`ols-access` scope)

#### Request body (MCPAppToolCallRequest)

| Field       | Type                                         | Required | Default | Description |
|-------------|----------------------------------------------|----------|---------|-------------|
| server_name | string                                       | yes      | --      | Name of the MCP server |
| tool_name   | string                                       | yes      | --      | The tool to invoke |
| arguments   | object                                       | no       | `{}`    | Arguments to pass to the tool |
| mcp_headers | object (server name -> header key/value map) | no       | null    | Client-provided auth headers |

Extra fields are rejected (422).

#### Response 200 (MCPAppToolCallResponse)

| Field              | Type                | Description |
|--------------------|---------------------|-------------|
| content            | array of ContentBlock | Content blocks returned by the tool |
| structured_content | object or null      | Structured data for the app UI, if the tool provides it |
| is_error           | boolean             | True if the tool call resulted in an error (default false) |

**ContentBlock** is one of:

- `{"type": "text", "text": "..."}` -- text content
- `{"type": "image", "data": "...", "mimeType": "..."}` -- image data (base64)
- `{"type": "audio", "data": "...", "mimeType": "..."}` -- audio data (base64)

#### Error responses

| Status | Condition |
|--------|-----------|
| 401    | Missing credentials, or required MCP server credentials not provided |
| 403    | Caller lacks permission |
| 404    | MCP server not configured |
| 500    | Communication failure with the MCP server |

---

### POST /v1/tool-approvals/decision

Submit an approval or rejection decision for a pending tool execution request. During streaming queries, certain tool calls may require explicit user approval before proceeding.

**Auth**: Required (`ols-access` scope)

#### Request body (ToolApprovalDecisionRequest)

| Field       | Type    | Required | Description |
|-------------|---------|----------|-------------|
| approval_id | string  | yes      | Unique identifier of the pending approval request |
| approved    | boolean | yes      | `true` to approve, `false` to reject |

Extra fields are rejected (422).

#### Response 200

Empty body (no content, 200 status).

#### Error responses

| Status | Condition |
|--------|-----------|
| 401    | Missing or invalid credentials |
| 403    | Caller lacks permission |
| 404    | No pending approval found for the given `approval_id` |
| 409    | The approval request has already been resolved (approved or rejected) |

---

### GET /v1/mcp/client-auth-headers

Discover which MCP servers require the client to supply authorization headers. Clients should use this to populate the `mcp_headers` field in query requests.

**Auth**: Required (`ols-access` scope)

#### Response 200 (MCPHeadersResponse)

| Field   | Type                         | Description |
|---------|------------------------------|-------------|
| servers | array of MCPServerHeaderInfo | Servers that need client-provided headers |

**MCPServerHeaderInfo:**

| Field            | Type            | Description |
|------------------|-----------------|-------------|
| server_name      | string          | Name of the MCP server |
| required_headers | array of string | Header names the client must provide |

---

### POST /authorized

Validate the caller's credentials and authorization to use the service. No `/v1` prefix.

**Auth**: Required (`ols-access` scope) -- the authentication check itself is the purpose of this endpoint.

#### Request body

None required.

#### Response 200 (AuthorizationResponse)

| Field              | Type          | Description |
|--------------------|---------------|-------------|
| user_id            | string (UUID) | The authenticated user's ID |
| username           | string        | The authenticated user's name |
| skip_user_id_check | boolean       | Whether strict user ID validation is bypassed |

#### Error responses

| Status | Condition |
|--------|-----------|
| 401    | Missing or invalid credentials |
| 403    | User is not authorized |
| 500    | Unexpected error during token review |

---

### GET /readiness

Kubernetes readiness probe. No authentication required. No version prefix.

Checks three subsystems: RAG index loaded (if configured), default LLM reachable and responsive, and conversation cache backend ready. All three must pass. The LLM readiness result is cached; once confirmed ready, subsequent calls may return the cached result for a configurable duration (`expire_llm_is_ready_persistent_state`).

#### Response 200 (ReadinessResponse)

| Field  | Type    | Description |
|--------|---------|-------------|
| ready  | boolean | Always `true` |
| reason | string  | `"service is ready"` |

#### Response 503 (NotAvailableResponse)

Structured error with cause indicating which subsystem is not ready: `"Index is not ready"`, `"LLM is not ready"`, or `"Cache is not ready"`.

---

### GET /liveness

Kubernetes liveness probe. No authentication required. No version prefix. Always succeeds if the process is running.

#### Response 200 (LivenessResponse)

| Field | Type    | Description |
|-------|---------|-------------|
| alive | boolean | Always `true` |

---

### GET /metrics

Prometheus/OpenMetrics metrics endpoint. No version prefix.

**Auth**: Required (`ols-metrics-access` scope) -- a separate permission scope from the main API.

#### Response 200

Plain text in Prometheus exposition format (`Content-Type: text/plain`).

---

## Streaming Format

The streaming query endpoint (`POST /v1/streaming_query`) supports two wire formats, selected by the `media_type` field in the request.

### JSON streaming (application/json)

Each event is delivered as a Server-Sent Events (SSE) data line:

```
data: {"event": "<event_type>", "data": {<payload>}}\n\n
```

#### Event catalog

**start** -- Sent once at the beginning of the stream.

```json
{"event": "start", "data": {"conversation_id": "<uuid>"}}
```

**token** -- One text token from the LLM response.

```json
{"event": "token", "data": {"id": 0, "token": "Some text"}}
```

The `id` field is a zero-based sequential counter shared across all `token` and `reasoning` events.

**reasoning** -- One token of chain-of-thought / reasoning content.

```json
{"event": "reasoning", "data": {"id": 0, "reasoning": "Let me think..."}}
```

**tool_call** -- The LLM is invoking a tool.

```json
{"event": "tool_call", "data": {"name": "tool_name", "args": {}, "id": "call_id", "type": "tool_call"}}
```

**approval_required** -- A tool call requires user approval before execution. The client must present the approval to the user and submit the decision via `POST /v1/tool-approvals/decision`.

```json
{"event": "approval_required", "data": {"approval_id": "...", ...}}
```

**tool_result** -- A tool execution has completed.

```json
{"event": "tool_result", "data": {"id": "call_id", "status": "success", "content": "...", "type": "tool_result", "round": 1}}
```

**skill_selected** -- A skill/capability was selected during processing.

```json
{"event": "skill_selected", "data": {"name": "skill_name"}}
```

**history_compression_start** -- Conversation history compression has begun (long conversations).

```json
{"event": "history_compression_start", "data": {}}
```

**history_compression_end** -- Conversation history compression has completed.

```json
{"event": "history_compression_end", "data": {}}
```

**end** -- Sent once after all processing is complete. Contains document references and token usage.

```json
{
  "event": "end",
  "data": {
    "referenced_documents": [{"doc_title": "...", "doc_url": "..."}],
    "truncated": false,
    "input_tokens": 123,
    "output_tokens": 456,
    "reasoning_tokens": 0
  },
  "available_quotas": {"QuotaLimiterName": 998000}
}
```

The `available_quotas` field is at the top level of the SSE payload, not inside `data`.

**error** -- An error occurred during stream processing. The stream ends after this event.

Prompt too long:
```json
{"event": "error", "data": {"status_code": 413, "response": "Prompt is too long", "cause": "details"}}
```

Generic error:
```json
{"event": "error", "data": {"response": "error summary", "cause": "details"}}
```

### Plain text streaming (text/plain)

No SSE framing. Output is concatenated directly.

| Event                      | Output format |
|----------------------------|---------------|
| start                      | Not emitted |
| token                      | Raw token text |
| reasoning                  | Raw reasoning text; a blank line (`\n\n`) separates the last reasoning token from the first text token |
| tool_call                  | `\nTool call: {json}\n` |
| approval_required          | `\nApproval request: {json}\n` |
| tool_result                | `\nTool result: {json}\n` |
| skill_selected             | `\nSkill selected: <name>\n` |
| history_compression_start  | `\nHistory compression start: {json}\n` |
| history_compression_end    | `\nHistory compression end: {json}\n` |
| end (with references)      | `\n\n---\n\n<title>: <url>` (one line per document) |
| end (no references)        | Empty string |
| error (prompt too long)    | `Prompt is too long: <details>` |
| error (generic)            | `<response>: <cause>` |

---

## Attachments

Query requests may include attachments -- supplementary content the user provides alongside their question.

### Allowed attachment types

`alert`, `api object`, `configuration`, `error message`, `event`, `log`, `stack trace`

### Allowed content types (MIME)

`text/plain`, `application/json`, `application/yaml`, `application/xml`

### Attachment schema

| Field           | Type   | Required | Description |
|-----------------|--------|----------|-------------|
| attachment_type | string | yes      | One of the allowed attachment types |
| content_type    | string | yes      | One of the allowed MIME content types |
| content         | string | yes      | The actual attachment content |

YAML attachments containing `kind` and `metadata.name` fields are treated as named Kubernetes resources.

---

## Query Modes

The `mode` field on query requests controls which system prompt and iteration limits are used. See `what/query-processing.md` for processing details and `what/agent-modes.md` for mode-specific behavior.

| Mode              | Default max iterations | Description |
|-------------------|------------------------|-------------|
| `ask`             | 5                      | General Q&A mode (default) |
| `troubleshooting` | 15                     | Troubleshooting mode with higher iteration limit for tool use |

---

## Standard Error Response Format

All error responses (4xx and 5xx) use one of two shapes:

**Structured error** (most endpoints):

```json
{
  "detail": {
    "response": "Human-readable summary of the error",
    "cause": "Technical detail or upstream error message"
  }
}
```

**Simple error** (401 and 403 on authenticated endpoints):

```json
{
  "detail": "Human-readable error message"
}
```

---

## Middleware

The service applies the following cross-cutting behaviors to all requests:

### Metrics and security headers

26. For every request to a known API route, measure and record response duration as a histogram metric labeled by path.
27. After the response is generated, increment a request counter metric labeled by path and HTTP status code. The `/metrics` endpoint itself is excluded from the counter.
28. Add security response headers to all responses except `/readiness`, `/liveness`, and `/metrics`:
    - `X-Content-Type-Options: nosniff` (always)
    - `Strict-Transport-Security: max-age=31536000; includeSubDomains` (only when TLS is enabled)

### Request/response logging

29. At debug log level, log the full request (client address, headers, body) and response (headers, body chunks for streaming) for every request.
30. Sensitive header values are redacted to `XXXXX`:
    - **Request headers**: `authorization`, `proxy-authorization`, `cookie`
    - **Response headers**: `www-authenticate`, `proxy-authenticate`, `set-cookie`
31. Logging of the `/metrics` endpoint can be suppressed via configuration (`suppress_metrics_in_log`).

---

## Configuration Surface

- `ols_config.default_provider` / `ols_config.default_model` -- default LLM provider and model when not specified in the request.
- `ols_config.authentication_config.module` -- authentication module (`k8s`, `noop`, `noop-with-token`).
- `ols_config.user_data_collection.feedback_disabled` -- disables feedback collection (POST /v1/feedback returns 403).
- `ols_config.user_data_collection.transcripts_disabled` -- disables transcript storage after queries.
- `ols_config.user_data_collection.feedback_storage` -- filesystem path for feedback JSON files.
- `ols_config.user_data_collection.transcripts_storage` -- filesystem path for transcript JSON files.
- `ols_config.logging_config.suppress_metrics_in_log` -- suppress debug logging for `/metrics` requests.
- `ols_config.expire_llm_is_ready_persistent_state` -- duration (seconds) to cache the LLM readiness check result. Negative or unset means cache indefinitely once ready.
- `ols_config.reference_content` -- when set, the readiness probe checks that the RAG index is loaded.
- `dev_config.disable_tls` -- disables TLS; affects whether HSTS header is added.
- `dev_config.enable_dev_ui` -- mounts an embedded Gradio UI at the application root.
- MCP server configuration -- defines available MCP servers, their URLs, timeouts, and header requirements. See `what/tools.md`.
- Quota limiter configuration -- defines per-user and per-cluster token quotas. See `what/quota.md`.

---

## Constraints

1. All conversation IDs must be valid SUIDs (UUID format). Invalid IDs are rejected with 400.
2. `provider` and `model` must both be present or both absent in query requests.
3. Feedback `sentiment` must be exactly `-1` or `1` if provided. At least one of `sentiment` or `user_feedback` must be set.
4. Attachment types and content types are restricted to the fixed allowed sets. No extensibility mechanism.
5. The streaming endpoint always returns HTTP 200 for the initial response; errors during generation are delivered as in-stream events.
6. MCP resource URIs must start with `ui://` and have a non-empty path component.
7. Tool approval decisions are idempotent in the forward direction only: once resolved, re-submitting returns 409.
8. Security headers are applied to all responses except health probes and metrics.
9. The `/metrics` endpoint uses a separate authorization scope (`ols-metrics-access`) from the rest of the API (`ols-access`).
10. Extra fields in request bodies are rejected (422) for query, MCP, and tool approval endpoints.

---

## Planned Changes

- [PLANNED: OLS-2682] Remove `/v1/query` endpoint. The streaming endpoint becomes the sole query interface.
- [PLANNED: OLS-2680] Add OpenAI `/responses` API compatibility layer.
- [PLANNED: OLS-2684] Remove client MCP headers (`mcp_headers` field).
