# Audit Logging

Implementation spec for compliance audit logging in lightspeed-service (OLS). Parent spec: `ols/.ai/spec/what/audit-logging.md` (authoritative for cross-repo requirements, event semantics, correlation contract, and OTel GenAI attribute reference).

## Separate Outbound Records [IMPLEMENTED: OLS-4192]

[Outbound request audit](outbound-request-audit.md) defines an additional metadata-only JSON log type for Python transport attempts. Its dedicated emitter is a narrowly scoped exception to rules 15–18 for this record type only. Existing compliance spans, full content capture, `user_id`, and exporters remain unchanged. They MUST NOT be copied into the new records, whose field allowlist excludes personal identity and content.

## Behavioral Rules

### Per-Request Traces

1. Each incoming HTTP request MUST generate a fresh, auto-generated OTel trace ID. The service MUST NOT use `conversation_id` as the trace ID.

2. `gen_ai.conversation.id` MUST be set as a span attribute on every span in the request trace. The value is the `conversation_id` UUID. Users query by `gen_ai.conversation.id` to see all requests in a conversation.

3. `user_id` (authenticated user identity from k8s token validation) MUST be set as a span attribute on every span in the request trace.

4. For multi-turn conversations, each request produces a separate trace with its own trace ID. All traces for the same conversation share `gen_ai.conversation.id` as a span attribute.

### Span Hierarchy

5. The service MUST create a root span `request.lifecycle` (kind `INTERNAL`) for each incoming request.

6. Child spans of `request.lifecycle`:

| Span Name | Kind | When | Key Attributes |
|---|---|---|---|
| `request.auth` | `INTERNAL` | User authentication | `user_id` |
| `request.rag` | `INTERNAL` | RAG chunk retrieval | Chunk count, source documents |
| `request.history` | `INTERNAL` | Conversation history load | Turn count, compressed (yes/no) |
| `chat {gen_ai.request.model}` | `CLIENT` | Each LLM turn | GenAI inference attributes (see below) |
| `execute_tool {gen_ai.tool.name}` | `INTERNAL` | Each tool call (child of `chat` span) | GenAI tool attributes (see below) |
| `request.store` | `INTERNAL` | Response storage | |

7. The root span `request.lifecycle` keeps its current name -- it encompasses the full HTTP request (auth, RAG, history, LLM turns, storage), not just the LLM call. Non-GenAI child spans (`request.auth`, `request.rag`, `request.history`, `request.store`) also keep their current names.

### GenAI Attributes on LLM Turn Span

8. Each `chat {gen_ai.request.model}` span MUST carry the following attributes:

| Attribute | Requirement | Description |
|---|---|---|
| `gen_ai.operation.name` | Required | `"chat"` |
| `gen_ai.request.model` | Required | Model name requested (e.g., `gpt-4o`, `claude-sonnet-4-6`) |
| `gen_ai.response.model` | Recommended | Actual model from provider response |
| `gen_ai.provider.name` | Required | Provider name (e.g., `openai`, `anthropic`, `watsonx`) |
| `gen_ai.usage.input_tokens` | Recommended | Input token count for this turn |
| `gen_ai.usage.output_tokens` | Recommended | Output token count for this turn |

9. OLS runs its own tool-calling loop (not an SDK agentic loop), so per-turn token counts are available and MUST be reported via `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` on each `chat {gen_ai.request.model}` span.

### GenAI Attributes on Tool Span

10. Each `execute_tool {gen_ai.tool.name}` span MUST carry the following attributes:

| Attribute | Requirement | Description |
|---|---|---|
| `gen_ai.operation.name` | Required | `"execute_tool"` |
| `gen_ai.tool.name` | Required | Tool name |
| `gen_ai.tool.call.id` | Recommended | Tool call ID from LLM response |
| `gen_ai.tool.type` | Recommended | `"function"` |

### MCP Attributes

11. When a tool is MCP-sourced, the `execute_tool {gen_ai.tool.name}` span MUST additionally carry:

| Attribute | Requirement | Description |
|---|---|---|
| `mcp.method.name` | Recommended | MCP method invoked (e.g., `tools/call`) |
| `mcp.session.id` | Recommended | MCP session identifier |
| `mcp.protocol.version` | Recommended | MCP protocol version |
| `gen_ai.tool.call.id` | Recommended | Tool call ID from MCP response |
| `network.transport` | Recommended | OTel OSI-layer values: `tcp` (mapped from MCP transports `streamable_http`/`sse`) or `pipe` (mapped from `stdio`) |

12. Non-MCP tools MUST carry all attributes from rule 10 (including `gen_ai.operation.name`) and MUST NOT add MCP-specific attributes from rule 11.

### Span Events

13. Text output from the LLM MUST be recorded as a `gen_ai.choice` span event attached to the `chat {gen_ai.request.model}` span. The event carries a `gen_ai.completion` attribute with the text content. This aligns with OTel GenAI Semantic Conventions.

14. Thinking/reasoning output from the LLM MUST be recorded as a `gen_ai.choice` span event with an additional `gen_ai.reasoning_content` attribute attached to the `chat {gen_ai.request.model}` span. When the model emits both completion and thinking content, they MAY be combined into a single `gen_ai.choice` event with both attributes.

### Tool Result Events

14b. Tool execution output MUST be recorded as a `tool.result` span event attached to the `execute_tool {gen_ai.tool.name}` span. The event carries a `success` attribute (boolean) and the raw, unwrapped tool output in an `output` attribute when the result passes inspection and audit logging is enabled.

14c. [PLANNED: OLS-3928] Tool-result inspection MUST conform to `openshift/ols/.ai/spec/what/tool-result-inspection.md`. Emit the `tool.result` event only after that individual result passes inspection. The event MUST include the raw, unwrapped output.

14d. The service MUST NOT emit a `tool.result` event or output attribute for a result rejected by inspection. Safe inspection-failure metadata MAY be recorded without rejected content.

14e. Audit capture is per result, even when tool calls execute concurrently. If one result passes inspection and a sibling result fails, the service MUST record the passing result with its raw, unwrapped output and MUST NOT record the rejected result. This audit record does not make the passing result eligible for model reinjection, a `tool_result` SSE event, or conversation/transcript storage; the all-or-nothing concurrent-round rule still applies to those destinations.

14f. The audit event MUST include the complete output from tool execution. Capture it before aggregate round-budget truncation. Preserve it separately from the model-facing copy.

### Content Capture Policy

14a. Completion and thinking span event attributes (`gen_ai.completion`, `gen_ai.reasoning_content`) and tool output (`output` on `tool.result` events) may contain PII or sensitive data. When audit logging is enabled, these content attributes MUST be captured at full fidelity. User-controlled content-capture settings are deferred; no `audit.capture_content` configuration is exposed.

### Single-Emission Rule

Rules 15–19 describe the existing content-bearing span audit. The separate metadata-only
LLM connectivity audit below is an explicit exception: it uses ordinary application logging
and does not copy any datum or content from existing audit spans/events.

15. Each audit-significant datum MUST be recorded exactly once, as an OTel span or span event. The stdout and OTLP exporters are two destinations for the same emission, not two separate emission paths.

16. Python `logging` MUST emit only developer-debugging messages and MUST NOT re-emit data that appears in spans or span events. This collapses any current dual-emission (structured JSON + OTel span) into a single path: OTel spans/events for audit (two exporters, one emission), standard logging for developer debugging only.

### Structured Log Format

17. Audit data MUST be emitted as OTel spans and span events. The stdout exporter serializes them as OTLP JSON -- the standard OTel wire format. There is no custom JSON format.

18. Two exporters on the same TracerProvider:
    - **OTLP exporter** -- sends spans to a trace backend when an endpoint is configured.
    - **Stdout exporter** -- serializes spans as OTLP JSON to stdout (always, when audit enabled). Python OTel SDK provides `opentelemetry.sdk.trace.export.ConsoleSpanExporter` natively.

19. The stdout exporter MUST NOT truncate span attributes or event attributes. Full fidelity is preserved. The stdout signal is the compliance record.

### Configuration

20. The service reads audit config from `olsconfig.yaml` (generated by the lightspeed-operator from `OLSConfig.spec.audit`):

```yaml
ols_config:
  audit:
    logging: Enabled          # Enabled (default) or Disabled; enabled is a derived property
    otel:
      endpoint: ""            # optional OTLP gRPC endpoint; no-op exporter when empty/absent
```

21. `audit.logging` controls audit eligibility. Its default is `Enabled`; `Disabled` suppresses new audit spans/events and connectivity attempts. Python code reads the derived `audit.enabled` property. An already-started connectivity attempt still receives its matching completion if the setting changes.

22. `audit.otel.endpoint` controls OTLP trace export. When set, the service configures an OTLP exporter with the given endpoint. CA trust for gRPC is handled via the `GRPC_DEFAULT_SSL_ROOTS_FILE_PATH` environment variable, set from `SSL_CERT_FILE` at startup. When endpoint is absent, a no-op OTLP exporter is used. The stdout exporter always emits regardless of whether an OTLP endpoint is configured.

### Span Hierarchy Diagram

23. Per-request trace structure:

```
request.lifecycle               [root, INTERNAL, gen_ai.conversation.id=<conv_id>, user_id=<uid>]
├── request.auth                [INTERNAL]
├── request.rag                 [INTERNAL]
├── request.history             [INTERNAL]
├── chat gpt-4o                 [CLIENT, repeats per LLM turn]
│   ├── execute_tool search     [INTERNAL, repeats per tool call]
│   │   └── (span events: tool.result)
│   └── (span events: gen_ai.choice)
└── request.store               [INTERNAL]
```

For multi-turn conversations, each request produces a separate trace. All traces for the same conversation share `gen_ai.conversation.id` as a span attribute. Query by `gen_ai.conversation.id` to see the full conversation.

## Metadata-only connectivity audit

`ols/utils/audit_support.py` emits `llm.request.started` / `llm.request.completed`,
`mcp.request.started` / `mcp.request.completed`, and `auth.request.started` /
`auth.request.completed` as
JSON messages at INFO through `ols.connection_audit`, with the existing service handlers,
formatting and level filtering. These are not OTLP log records; the formatted log line may
have the normal application-log prefix. No new telemetry provider/exporter/configuration or
Collector dependency is introduced. Existing `ols/utils/otel.py` behavior remains unchanged.

Generated `attempt_id` identifies one owned native HTTP attempt. A lock-protected in-memory
dictionary retains only active safe metadata and a monotonic start; atomic removal is the
once-only completion gate. Retries have separate attempt IDs and one logical-call ID per
dispatch. Generate emitted logical-call IDs locally; never copy caller-supplied runnable or
native SDK run IDs into audit records. Use only private transient native object-identity keys
to recognize nested dispatch, including async-to-sync fallback; never emit those keys. Keep
independent invocations separate even when they reuse a native run ID. Preserve native SDK
run IDs, callbacks and validation timing unchanged. Local trace/span IDs are optional
write-only links to existing tracing, not matching keys or backend queries. Missing trace context is valid. A process crash may leave a start
without completion; live attempts are not expired or declared complete by an observer timeout.

Observe native EOF, close, failure and cancellation, not just response headers. HTTP success
is not model, tool, JSON-RPC or credential semantic success. Early close, including SSE DONE
before observed physical EOF, is conservatively incomplete. Preserve native async-generator finalization and client/stream
ownership. A readiness future timeout does not complete its still-running HTTP request.
Logging/cleanup failures must not replace native errors or retain completed dictionary entries.

Record fields (schema version 1; omit unknown optional values):

- Common: `schema_version`, `event`, `timestamp`, `attempt_id`, `logical_call_id`, `operation`,
  `streaming`, optional `trace_id`/`span_id`.
- LLM identity only: `provider`, `hosting`, optional audit-safe `model`.
- MCP identity only: configured `mcp_server`, `mcp_transport` (`streamable_http` or `sse`).
- Credential identity only: controlled `provider` (`azure`, `google`, `aws`), operation `token`.
- Authority/measurement: `destination_scheme`, `destination_host`, `destination_port`, optional
  `target_scheme`/`target_host`/`target_port` for a proxy, `request_body_bytes`, `timeout_ms`.
- Completion: `outcome` (`success`/`failure`/`cancelled`), `duration_ms`, `http_status`,
  `response_headers_ms`, `response_complete`, `response_body_bytes`, `failure_category`.
- Native TLS evidence only: `decision`, `decision_control`, `deny_reason`.

For Vertex, project recognized publisher-model resource identifiers to their public model
name. Omit `model` for private model/endpoint resources or unrecognized resource/URI forms;
never retain their project/path in audit metadata. Keep the original SDK model argument and
ordinary model names (including other providers' qualified names) unchanged.

Controlled failure categories: `dns`, `connect`, `tls`, `timeout`, `transport`, `http_error`,
`stream_error`, `incomplete_response`, `cancelled`. Operations: `agent`, `tool_result_classifier`,
`history`, `health`, or fallback `model`. Provider/hosting pairs: OpenAI `openai/openai`, Azure
`openai/azure`, vLLM `openai/rhoai_vllm` or `openai/rhelai_vllm`, Anthropic `anthropic/anthropic`,
Vertex `gemini/gcp.vertex_ai` or `anthropic/gcp.vertex_ai`, Bedrock `anthropic/bedrock` or
`openai/bedrock`, Watsonx `watsonx/watsonx`.

The loader observes the original constructed model's native transports through consolidated
`ols/utils/audit_support.py` adapters. Preserve HTTPX/HTTPX2/aiohttp/urllib3/backend
selection, proxy mounts, pooling, TLS, retries, backpressure, messages, callbacks and token
accounting. Retain the native model name, qualified name and module on the dispatch subclass
so LangChain naming, serialization/callback identity and cache keys remain native even with
audit logging disabled. Reuse existing native cache entries; cache hits emit no HTTP attempts.
IBM physical observation is below its retry loop and follows native TLS pool
recreation. SDK-private hooks are version-sensitive and need native boundary regression tests.
Unknown custom IBM transports without native pools are left unchanged, not replaced.

Never include prompts, payloads, headers, credentials, raw exception text, full URLs, paths,
queries or user/conversation/account/tenant/project/role/session identities. Count bytes only
as consumed. Do not copy span content or unrelated application logs. Constructor/setup,
metadata/token-count helpers and credential traffic are not LLM or MCP attempts. Static
credentials, cache hits, preflight failures, local signing, incoming auth, Kubernetes,
embeddings and remotely owned HTTP do not generate synthetic attempts. Live cloud/cluster
qualification remains deferred. No guarantee of remote delivery, downstream indexing or
five-second searchable availability is asserted.

### Credential ownership and native acquisition

Port the sandbox `AuthRequestMetadata` / `auth_request` pattern rather than credential policy.
A credential scope uses its own logical-call ID and optional originating trace snapshot;
nested same-provider scopes share that identity and restore the previous LLM/MCP scope.
Only actual observed HTTP emits `auth.request.*`, including physical native retries.

- Azure: decorate the original `ClientSecretCredential` token methods and its existing lazy
  requests transport. Observe native acquisition, including MSAL discovery HTTP. Preserve
  the service token cache, scopes, native credential/cache, pipeline, retries and transport.
- Google Vertex (Gemini and Anthropic): decorate the native credentials returned by
  `load_vertex_credentials`. Observe the requests session supplied to native `refresh`,
  including Google async SDK worker refresh. Keep both supported JSON credential types,
  JWT grants, native caches and local self-signing policy unchanged. Decorate native proxy
  managers as selected by the original requests adapter; do not construct a session.
- AWS: decorate the original STS client's connection-manager selection before the service's
  existing `assume_role`. Keep eager acquisition, role/session parameters, signing and returned
  native boto3 session unchanged; do not copy sandbox's lazy refresh credential policy.

This sandbox port does **not** cover Watsonx IAM acquisition/refresh or arbitrary MCP OAuth
helper exchanges. Watsonx's first IAM exchange occurs during SDK model construction;
its original ownership/construction remains unchanged. Those are explicit coverage gaps,
not synthetic successes. Unsupported custom credential/transport shapes remain unchanged.

### MCP ownership and purpose

`gather_mcp_tools` decorates native HTTP client factories without mutating caller connection
configuration. Existing custom factories still receive their headers, timeout and auth
objects and retain their client, TLS/proxy policy, pool, mounts and ownership. Discovery
and the native tool interceptor carry controlled purposes into client construction. MCP
Apps observe their existing native HTTP clients for resource reads and tool calls.

MCP operations are `discovery`, `tool`, `resource`, or fallback `connection`. They describe
the owning logical operation, not a JSON-RPC method inferred by inspecting request content.
The client captures immutable server/purpose/logical-call/trace metadata once; native SDK
background tasks and body consumers do not borrow unrelated LLM or later caller context.
The `streaming` flag follows the native HTTP `send(stream=...)` mode for each request.

Count actual HTTP traffic, including initialization, notifications, discovery pagination,
SSE GET/reconnection, authentication retries of the original MCP request, and session
termination. Long-lived SSE responses finish only at observed EOF, close, failure or
cancellation, not when one RPC result arrives. Native auth-flow helper requests are not
MCP attempts. Stdio and server-owned downstream HTTP are not instrumented. Do not retain
MCP session IDs, JSON-RPC IDs, tool-call IDs, tool names, arguments, results or resource URIs.
An HTTP 200 carrying a tool error can still be a successful physical HTTP attempt; existing
tool error handling and safety decisions remain unchanged.

## Cross-References

- `observability.md` -- existing Prometheus metrics and `gen_ai.*` histogram metrics
- `query-processing.md` -- query pipeline stages where spans are created
- `tools.md` -- tool execution flow, MCP integration
- `auth.md` -- k8s token validation where user identity is extracted
- Parent spec `ols/.ai/spec/what/audit-logging.md` -- OTel GenAI attribute reference, correlation model, single-emission rule
