# Audit Logging

Implementation spec for compliance audit logging in lightspeed-service (OLS). Parent spec: `ols/.ai/spec/what/audit-logging.md` (authoritative for cross-repo requirements, event semantics, correlation contract, and OTel GenAI attribute reference).

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
| `gen_ai.request.model` | Required | Model name requested (e.g., `gpt-4o`, `claude-sonnet-4-20250514`) |
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
| `network.transport` | Recommended | `stdio` or `sse` |

12. Non-MCP tools MUST carry all attributes from rule 10 (including `gen_ai.operation.name`) and MUST NOT add MCP-specific attributes from rule 11.

### Span Events

13. Text output from the LLM MUST be recorded as a `gen_ai.choice` span event attached to the `chat {gen_ai.request.model}` span. The event carries a `gen_ai.completion` attribute with the text content. This aligns with OTel GenAI Semantic Conventions.

14. Thinking/reasoning output from the LLM MUST be recorded as a `gen_ai.choice` span event with an additional `gen_ai.reasoning_content` attribute attached to the `chat {gen_ai.request.model}` span. When the model emits both completion and thinking content, they MAY be combined into a single `gen_ai.choice` event with both attributes.

### Content Capture Policy

14a. Completion and thinking span event attributes (`gen_ai.completion`, `gen_ai.reasoning_content`) contain LLM output that may include PII or sensitive data. Recording these attributes MUST be opt-in, controlled by an `audit.capture_content` configuration flag (default: `false`). When `capture_content` is `false`, `gen_ai.choice` events are still emitted but the content attributes are omitted. This aligns with the OTel GenAI semantic convention requirement level of Opt-In for content attributes.

### Single-Emission Rule

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
    enabled: true             # default: true (audit on even if section is absent)
    capture_content: false    # default: false; opt-in to record LLM output in span events
    otel:
      endpoint: ""            # optional OTLP gRPC endpoint; no-op exporter when empty/absent
      tls_mode: Secure        # Secure (default) | Insecure
```

21. `audit.enabled` controls audit emission. Defaults to `true` -- when the config section is absent or the field is not set, audit logging is on. Set to `false` to suppress all audit spans and events.

22. `audit.otel.endpoint` controls OTLP trace export. When set, the service configures an OTLP exporter with the given endpoint. `tls_mode` defaults to `Secure`; set to `Insecure` for plaintext gRPC. When endpoint is absent, a no-op OTLP exporter is used. The stdout exporter always emits regardless of whether an OTLP endpoint is configured.

### Span Hierarchy Diagram

23. Per-request trace structure:

```
request.lifecycle               [root, INTERNAL, gen_ai.conversation.id=<conv_id>, user_id=<uid>]
├── request.auth                [INTERNAL]
├── request.rag                 [INTERNAL]
├── request.history             [INTERNAL]
├── chat gpt-4o                 [CLIENT, repeats per LLM turn]
│   ├── execute_tool search     [INTERNAL, repeats per tool call]
│   └── (span events: gen_ai.choice)
└── request.store               [INTERNAL]
```

For multi-turn conversations, each request produces a separate trace. All traces for the same conversation share `gen_ai.conversation.id` as a span attribute. Query by `gen_ai.conversation.id` to see the full conversation.

## Cross-References

- `observability.md` -- existing Prometheus metrics and `gen_ai.*` histogram metrics
- `query-processing.md` -- query pipeline stages where spans are created
- `tools.md` -- tool execution flow, MCP integration
- `auth.md` -- k8s token validation where user identity is extracted
- Parent spec `ols/.ai/spec/what/audit-logging.md` -- OTel GenAI attribute reference, correlation model, single-emission rule
