# Classic OLS Outbound Request Audit

[PLANNED: OLS-4192] Python-level outbound request instrumentation for lightspeed-service. The workspace specification `ols/.ai/spec/what/outbound-request-audit.md` is authoritative for the shared version-1 JSON schema, lifecycle, privacy boundary, and scope. This child specification defines Classic coverage and integration requirements; it does not claim instrumentation is implemented.

## Intent

Operators can determine which application destination the service attempted to contact, correlate the attempt to a request, and see the observed outcome without collecting prompts, credentials, response bodies, or personal user identity in the new records. Application traffic includes LLM, remote MCP, retrieval, and their SDK credential exchanges. Kubernetes API request auditing is excluded as a separate infrastructure concern. These are metadata-only request records, not replacements for content-bearing compliance audit spans.

## Behavioral Rules

### Coverage

1. Instrument production Python/SDK client requests for every supported LLM provider. Cover sync and async invocation, streaming/non-streaming calls, shaping, inspection, model/embedding calls where used, and SDK credential/token acquisition. A LangChain model callback alone is not transport coverage: SDK retries and additional requests must remain visible.
2. Instrument remote MCP initialization, discovery, resource operations, tool calls, and MCP apps request paths. Set `gen_ai.tool.name` only during a known specific MCP tool call. Discovery, resource fetches, and ordinary LLM calls MUST NOT inherit the last tool name. Local stdio IPC itself is not an outbound network request; a spawned MCP process's network activity is outside this Python instrumentation scope.
3. Kubernetes API requests, including authentication TokenReview/SubjectAccessReview and other service-initiated cluster API calls, are infrastructure traffic and MUST NOT enter this application-level audit stream. Their auditing is a separate concern, regardless of their Python client implementation. This does not exclude the service's request to a remote MCP tool that accesses Kubernetes, or cloud credential exchanges for in-scope application clients.
4. Instrument Python retrieval/HTTP requests, including Solr/OKP and startup discovery requests. Local FAISS access is not network activity. PostgreSQL wire traffic and audit/telemetry export are excluded under the shared contract; do not infer universal socket coverage.
5. In-scope application requests during startup or background work remain covered even without an inbound request or conversation. Infrastructure requests do not become in scope merely because they occur in the same lifecycle. The component MUST establish valid OTel attempt context without inventing conversation or user identifiers.

### Record Production and Correlation

6. Emit exactly one `outbound.request.started` / `outbound.request.finished` pair per observable transport attempt under the shared schema. Apply separate attempt IDs to SDK/client retries and redirect hops. Do not double-count requests through both a common client factory and a provider wrapper.
7. Use a dedicated JSON stdout audit emitter, not a serialization of existing `chat`, tool, authentication, or request-lifecycle spans. This is an explicit exception to [audit-logging.md](audit-logging.md)'s span-only/Python-logging restrictions for this new record type only. No generic span attributes, personal `user_id`, automatic exception events, or trace baggage may be copied into the record.
8. Carry the active trace/span context plus the existing validated conversation UUID when available. Capture them before transport invocation and retain them through completion, including synchronous SDK work dispatched from async request processing. Concurrent requests and parallel MCP calls MUST NOT leak correlation or tool identity into each other.
9. Honor `ols_config.audit.enabled` and its existing default from [audit-logging.md](audit-logging.md). No new config field is introduced. Disabled audit suppresses the new records without changing request execution, independent tracing, transcripts, or feedback collection. Enabled records MUST emit without an OTLP endpoint and independently of trace sampling.
10. Request outcome is observed transport/protocol outcome, not an allow/deny decision. An in-scope LLM or MCP HTTP 403 is `http_error`; a refused connection is a transport failure, not proof of NetworkPolicy rejection. Keep error categories controlled and exclude raw messages/tracebacks.
11. Finish streaming attempts on EOF, explicit close, cancellation, or error, not upon response headers. Closing a response early is a completed transport lifetime, not proof the entire application response was consumed; cancellation/error MUST retain their observed outcome. Do not force draining, buffering, or retries for auditing. Count request-body bytes only when a complete transmitted size is naturally measurable.

### Compatibility and Constraints

12. Preserve existing provider parameters, SDK behavior, TLS and CA handling, proxying, authentication, retries, timeouts, stream delivery, MCP apps behavior, and the service's public API. Provider integrations that do not use the shared httpx factory still require transport-appropriate hooks; do not force them onto a different SDK just for audit.
13. Preserve existing full-fidelity compliance events, their `user_id` contract, inspection behavior, and exporters. The no-personal-identity/no-content policy applies to the new outbound records, not a retroactive rewrite of existing approved evidence. Selecting outbound audit logs alone MUST not require including those content-bearing records.
14. Preserve Gateway audit when deployed. Do not attribute remote MCP server downstream traffic to the service or assume Gateway covers service-side connection failures.
15. No operator API, deployment component, external backend, network-observer integration, or five-second searchability requirement is introduced. Subprocess connections are excluded; complementary sandbox network observation is tracked separately by OLS-4848. Kubernetes logging supplies source pod metadata; the service MUST NOT query the API server just to enrich its logs.
16. Emitter failures MUST not change application results, exception propagation, or cancellation. Prevent audit recursion and duplicate emission through root logger propagation. Hook installation MUST be idempotent across initialization and config reload; retries must not reinstall hooks.

## Implementation Boundaries and Inventory

These are integration locations to inspect, not a mandate to implement one global monkey patch. The implementation PR MUST record the supported route/transport matrix and prove each entry with tests.

| Client family | Existing integration locations | Required evidence |
| --- | --- | --- |
| Common LLM HTTP clients | `ols/src/llms/providers/provider.py` | Sync/async requests, streaming close, existing TLS/proxy behavior, no duplicate hooks |
| Provider-specific SDK clients | `ols/src/llms/providers/` including Bedrock-specific clients | All supported provider routes; non-httpx transports and credential requests; SDK retries |
| MCP sessions and tools | `ols/src/tools/tools.py` and related session helpers | Session/discovery versus per-tool context; concurrent calls; SSE/streamable HTTP request lifetimes |
| MCP apps HTTP | `ols/app/endpoints/mcp_apps.py` | Resource/proxy requests without leaking paths, client headers, or bodies |
| Kubernetes auth client — excluded | `ols/src/auth/k8s.py` | TokenReview/SubjectAccessReview and other cluster API traffic remain outside this application audit stream |
| Retrieval | `ols/src/rag_index/solr_support.py` | Hybrid query and startup/discovery requests; graceful-degradation behavior preserved |
| Audit configuration and existing compliance logging | `ols/utils/audit_logger.py`, `ols/app/models/config.py` | Existing enablement, independent safe emitter, no change to compliance content |

## Verification — Required for Implementation

No outbound-audit test coverage is claimed by this specification-only change.

- Unit tests: validate every schema field, optional-field omission, safe destination parsing, bounded labels, and controlled error mapping. Assert the full serialized record excludes planted secrets and personal identities, including inherited `user_id` from a parent span.
- Client tests: sync/async success, HTTP 401/403/429/5xx, connect/DNS/TLS/timeout errors, non-HTTP SDK failure, SDK retry, redirect, nested hooks, and complete/unknown body-size handling.
- Infrastructure exclusion tests: Kubernetes authentication/API calls emit no application outbound records, including when they share transport libraries with instrumented application requests. Existing authentication behavior remains unchanged.
- Streaming tests: normal EOF, explicit early close, transport failure during iteration, cancellation before headers and during body consumption, and exact-once finish without consuming additional bytes.
- Context tests: two simultaneous conversations, parallel different MCP tools, discovery after a tool call, synchronous/threaded clients, and startup/background requests with missing conversation context.
- Gate tests: audit enabled/disabled, no tracing endpoint, unsampled valid trace context, logger failure, recursive export suppression, and repeated initialization/config reload.
- Regression tests: existing compliance spans/content/identities, inspection, transcripts, metrics, TLS/proxies, MCP apps, Gateway integration when present, and original application exceptions remain unchanged.
- Deployment verification: parse stdout JSON and confirm pod enrichment/selection of only `outbound.request.*` records through OpenShift cluster logging or an equivalent supported forwarding fixture. This is integration compatibility evidence, not a delivery or search-latency SLA.

## References

- Workspace `ols/.ai/spec/what/outbound-request-audit.md` — shared contract and OLS-4848 boundary.
- [Audit logging](audit-logging.md) — existing compliance evidence and configuration.
- [LLM providers](llm-providers.md), [tools](tools.md), [MCP apps](mcp-apps.md), [authentication](auth.md), [RAG](rag.md) — existing client behavior.
