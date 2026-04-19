# System Overview

OpenShift LightSpeed (OLS) is an AI-powered assistant service that answers
natural-language questions about OpenShift and related Red Hat products by
orchestrating LLM calls, RAG-augmented prompts, multi-turn conversation
history, and live cluster introspection via MCP tools. [PLANNED: OLS-2743]
The service is being rebranded to "Red Hat OpenShift Intelligent Assistant."

## Behavioral Rules

### Identity and Scope

1. The service is a REST API server (currently named "OpenShift LightSpeed")
   that processes user queries, orchestrates LLM interactions, and returns
   AI-generated responses. It does NOT include the operator, the console UI
   plugin, or the RAG content build pipeline. [PLANNED: OLS-2743] The
   service name will change to "Red Hat OpenShift Intelligent Assistant."

2. The service is product-agnostic at its core. It can serve as an AI
   assistant for any Red Hat product by changing the system prompt
   (configurable via `ols_config.system_prompt_path`), loading
   product-specific RAG content, and adjusting topic guardrails. The service
   detects its deployment context (e.g., OpenStack Lightspeed) and applies
   the appropriate system prompt automatically.

3. The service exposes a versioned REST API under the `/v1` prefix. All
   query, feedback, conversation, MCP, and tool-approval endpoints are
   versioned. Health and metrics endpoints are unversioned.

### Chat and Query Processing

4. The service accepts a natural-language question from a user and returns
   an AI-generated answer. The service enforces topic guardrails so the LLM
   responds only to questions relevant to the configured product domain.
   Off-topic or harmful queries are rejected.

5. Two query modes control the system prompt and tool-calling depth:
   **ASK** (general product questions) and **TROUBLESHOOTING** (diagnostic
   and remediation questions with more tool-calling iterations allowed).
   The mode is specified per request.

6. Responses are streamed token-by-token to the client as the LLM generates
   them, reducing perceived latency. A non-streaming endpoint is also
   available that returns the complete response.

7. Before sending a question to the LLM, the service retrieves relevant
   chunks from a pre-built product documentation index (RAG) and injects
   them into the prompt. Retrieved chunks must exceed a similarity threshold
   to be included. The RAG index is versioned per product release so answers
   match the user's cluster version.

8. The service maintains per-user, per-conversation history. History is
   injected into subsequent prompts for multi-turn context. History is
   truncated to fit within the model's context window. When history
   compression is enabled (configurable via
   `ols_config.history_compression_enabled`), the service summarizes
   older history to reduce token consumption.

9. Users cannot access another user's conversation history. Conversation
   cache entries are keyed by both conversation ID and authenticated user
   identity.

10. Users can attach contextual data to their questions. Supported
    attachment types: alert, api object, configuration, error message,
    event, log, stack trace. Supported content types: text/plain,
    application/json, application/yaml, application/xml. Attachments are
    included in the prompt so the LLM can reason about the user's specific
    situation.

### LLM Providers

11. The service supports the following LLM provider types: OpenAI, Azure
    OpenAI (including Entra ID / service-principal authentication),
    WatsonX, RHOAI vLLM, RHEL AI vLLM, Google Vertex (Anthropic), and
    Google Vertex. The administrator configures providers and models via
    `llm_providers` in the configuration file. [PLANNED: OLS-2521] Google
    Gemini model support is being added. [PLANNED: OLS-2776] Anthropic as a
    direct LLM provider is planned. [PLANNED: OLS-1680] AWS Bedrock as an
    LLM provider is planned. [PLANNED: OLS-1660] Llama Stack integration
    is in progress.

12. A default provider and model are configured at the service level via
    `ols_config.default_provider` and `ols_config.default_model`. The
    service abstracts provider differences so all capabilities work
    uniformly regardless of backend.

### MCP Tools and Skills

13. The service acts as an MCP (Model Context Protocol) client, connecting
    to externally deployed MCP servers to provide tools for querying live
    cluster state. MCP servers are configured via `mcp_servers.servers` in
    the configuration file, each with a name, URL, optional timeout, and
    authorization headers.

14. MCP authorization headers support three resolution modes: a static file
    path, the placeholder `kubernetes` (uses the user's Kubernetes token),
    or the placeholder `client` (uses a client-provided header value).

15. Tool execution can require user approval before running. The approval
    strategy is configurable via `ols_config.tools_approval.approval_type`:
    `never` (no approval needed), `always` (all tool calls require
    approval), or `tool_annotations` (approval based on per-tool
    annotations). Approval requests time out after a configurable period
    (default 600 seconds).

16. When MCP tools are configured, the LLM may request tool calls during
    query processing. The service executes tool calls against MCP servers
    and feeds results back to the LLM in an iterative loop. The maximum
    number of iterations is mode-dependent: ASK mode has a lower default
    limit than TROUBLESHOOTING mode. A per-request override is available
    via `ols_config.max_iterations`.

17. Tool filtering can be enabled (via `ols_config.tool_filtering`) to
    select the most relevant tools for a given query using hybrid
    dense/sparse retrieval, reducing noise and improving tool-call accuracy.

18. The service supports skills -- configurable prompt strategies loaded
    from a skills directory (configurable via `ols_config.skills.skills_dir`).
    Skills are matched to user queries using hybrid RAG retrieval with a
    configurable similarity threshold. When a skill matches, its specialized
    prompt and tool combination are used instead of the generic chat flow.

### Token Budget and Context Window

19. Each request operates within a token budget derived from the model's
    context window size (configurable per model, default 128,000 tokens).
    The budget is partitioned across system prompt, RAG context, history,
    tool outputs, and response tokens.

20. When MCP tools are configured, a configurable fraction of the context
    window is reserved for tool outputs (configurable via model-level
    `tool_budget_ratio`, default 25%, range 10%-60%). Within each
    tool-calling round, a cap limits how much of the remaining tool budget
    can be consumed (configurable via `ols_config.tool_round_cap_fraction`,
    default 60%, range 30%-80%).

21. The maximum tokens reserved for the LLM response is configurable per
    model (default 4,096 tokens).

### Quota Management

22. Administrators can define usage quotas per user and per cluster. Quotas
    limit the number of LLM tokens consumed over a configurable time
    window. Quota configuration is under `ols_config.quota_handlers` and
    requires a PostgreSQL storage backend and a scheduler period.

23. When a user or cluster exceeds their token quota, the service rejects
    the request.

### Data Redaction

24. Before sending prompts to an LLM provider, the service applies
    configurable regex-based query filters to redact sensitive data.
    Filters are defined via `ols_config.query_filters`, each specifying a
    name, regex pattern, and replacement string.

### Authentication and Authorization

25. By default, users authenticate via Kubernetes token validation. The
    service verifies the user's cluster API token on each request. The
    authentication module is configurable via
    `ols_config.authentication_config.module`. Supported modules: `k8s`,
    `noop` (disabled), `noop-with-token` (testing with token). The default
    is `k8s`.

26. Authorization is controlled through standard Kubernetes RBAC. The
    service checks access against a virtual path (`/ols-access`) using the
    authenticated user's token.

27. All API endpoints except health and metrics require authentication.
    Health endpoints (`/readiness`, `/liveness`) and the `/metrics` endpoint
    are unauthenticated.

### TLS and Security

28. All endpoints are TLS-encrypted in production. TLS certificates are
    configured via `ols_config.tls_config`. TLS can be disabled for
    development via `dev_config.disable_tls`.

29. The service respects the cluster-level TLS security profile for cipher
    suite and minimum TLS version selection (configurable via
    `ols_config.tlsSecurityProfile`). The `OldType` profile is rejected;
    minimum allowed TLS version is 1.2.

30. Security response headers (X-Content-Type-Options, Strict-Transport-
    Security) are added to all API responses except health and metrics
    endpoints.

31. HTTP request and response logs redact sensitive headers (authorization,
    proxy-authorization, cookie, www-authenticate, proxy-authenticate,
    set-cookie).

32. Connections to LLM providers use TLS. Credentials are read from file
    paths specified in the provider configuration. Extra CA certificates
    can be configured via `ols_config.extra_ca`.

### User Feedback

33. Users can submit feedback (positive/negative with optional free-text)
    on responses. Feedback is stored locally on the cluster filesystem.
    Feedback can be forwarded to Red Hat via the Insights telemetry
    pipeline. Feedback collection is configurable via
    `ols_config.user_data_collection.feedback_disabled` (disabled by
    default).

34. When feedback is disabled, the feedback endpoint returns HTTP 403.

### Conversation Management

35. Conversation history is stored in a persistent cache. Two storage
    backends are supported: in-memory (suitable for development and
    single-replica deployments, with configurable max entries and LRU
    eviction) and PostgreSQL (required for production and multi-replica
    deployments). Configured via `ols_config.conversation_cache.type`.

36. The service exposes endpoints for listing, retrieving, updating (rename),
    and deleting conversations. All conversation operations are scoped to
    the authenticated user.

### Observability

37. The service exposes Prometheus-compatible metrics at `/metrics`
    covering: REST API call counts by path and status code, response
    duration histograms by path, LLM call counts, LLM call failure counts,
    LLM tokens sent and received, LLM reasoning tokens, and provider/model
    configuration gauge.

38. Conversation transcripts can be optionally recorded for quality
    analysis. Configured via
    `ols_config.user_data_collection.transcripts_disabled` (disabled by
    default).

39. Configuration status (summary of provider, model, and enabled features)
    can be reported for aggregate product analytics when feedback or
    transcript collection is enabled.

### Health Probes

40. The service exposes `/readiness` and `/liveness` health endpoints.
    Liveness always returns healthy if the process is running. Readiness
    checks three conditions: RAG index is loaded (if configured), LLM
    provider is reachable and responding, and conversation cache is ready.
    All three must pass for the service to report ready.

41. The LLM readiness check result is cached to avoid repeated probe calls
    to the LLM. The cache expiration is configurable via
    `ols_config.expire_llm_is_ready_persistent_state`.

### Development Mode

42. A developer configuration section (`dev_config`) allows: enabling an
    embedded Gradio UI (`enable_dev_ui`), disabling authentication
    (`disable_auth`), disabling TLS (`disable_tls`), overriding the system
    prompt at request time (`enable_system_prompt_override`), configuring a
    Pyroscope profiling URL (`pyroscope_url`), and binding to localhost
    (`run_on_localhost`).

## Configuration Surface

All configuration is read from `olsconfig.yaml` (path overridden via the
`OLS_CONFIG_FILE` environment variable). The top-level structure:

| Section | Purpose |
|---|---|
| `llm_providers` | List of LLM provider configurations (name, type, URL, credentials, models) |
| `ols_config.default_provider` / `default_model` | Default LLM backend for queries |
| `ols_config.reference_content` | RAG index paths and embeddings model path |
| `ols_config.conversation_cache` | Cache backend selection (memory or postgres) and settings |
| `ols_config.authentication_config` | Auth module, K8s API URL, CA cert, TLS skip |
| `ols_config.tls_config` | TLS certificate and key paths |
| `ols_config.tlsSecurityProfile` | Cluster TLS profile (type, minTLSVersion, ciphers) |
| `ols_config.query_filters` | Regex-based redaction filters (name, pattern, replace_with) |
| `ols_config.quota_handlers` | Token quota configuration (storage, scheduler, limiters) |
| `ols_config.user_data_collection` | Feedback and transcript collection toggles and storage paths |
| `ols_config.logging_config` | Log levels for app, libraries, and uvicorn; metric log suppression |
| `ols_config.system_prompt_path` | Path to custom system prompt file |
| `ols_config.max_iterations` | Override default max tool-calling iterations |
| `ols_config.history_compression_enabled` | Enable/disable history summarization (default: true) |
| `ols_config.proxy_config` | HTTP/HTTPS proxy and no-proxy settings |
| `ols_config.extra_ca` | Additional CA certificate file paths |
| `ols_config.tool_filtering` | Tool filtering via hybrid RAG (embed model, alpha, top_k, threshold) |
| `ols_config.tools_approval` | Tool execution approval strategy and timeout |
| `ols_config.skills` | Skills directory, embed model, alpha, threshold |
| `ols_config.tool_round_cap_fraction` | Max fraction of tool budget usable per round |
| `mcp_servers.servers` | List of MCP server configurations (name, URL, timeout, headers) |
| `dev_config` | Development-only toggles (UI, auth, TLS, profiling, localhost) |

## Constraints

1. **Service-only scope.** This specification covers the lightspeed-service
   Python application only. The operator (lightspeed-operator), console
   plugin (lightspeed-console), RAG content pipeline (lightspeed-rag-content),
   and evaluation framework (lightspeed-evaluation) are separate components.

2. **Conversation isolation.** Conversations are scoped to the authenticated
   user identity. No cross-user access is permitted.

3. **Context window hard limit.** All prompt components (system prompt, RAG
   context, history, attachments, tool outputs, response budget) must fit
   within the model's context window. The service truncates history and RAG
   context as needed; if the prompt still exceeds the limit after truncation,
   the request is rejected with HTTP 413.

4. **TLS minimum version.** The `OldType` TLS security profile is rejected.
   Minimum allowed TLS protocol version is 1.2.

5. **RAG is read-only.** The service loads and queries pre-built RAG indexes
   at startup. It does not build, modify, or update indexes at runtime.

6. **MCP servers are external.** The service connects to MCP servers as a
   client. MCP servers are independently deployed, configured, and secured.

7. **Authentication default.** When no authentication module is configured,
   the default is Kubernetes token validation (`k8s`).

8. **Supported architectures.** The service supports x86_64 and ARM
   (aarch64).

9. **FIPS readiness.** The service is designed for FIPS -- it uses
   FIPS-validated cryptographic modules, follows FIPS guidelines, and is
   deployable on FIPS-enabled OpenShift clusters.

10. **Disconnected operation.** All features work without internet access,
    provided the LLM provider is reachable from the cluster.

11. **HCP compatibility.** The service is compatible with OpenShift Hosted
    Control Planes deployments.

## Planned Changes

| Jira Key | Summary |
|---|---|
| OLS-2743 | Rebranding to "Red Hat OpenShift Intelligent Assistant" |
| OLS-2894 | Autonomous, policy-driven AI agents for OpenShift (TP: OCP 5.0) |
| OLS-1660 | Use of Llama Stack in OLS |
| OLS-2521 | Support Google Gemini model in OLS |
| OLS-2776 | Support Anthropic as a direct LLM provider |
| OLS-1680 | Support AWS Bedrock as LLM provider |
| OLS-2823 | Per-user LLM provider API keys |
| OLS-1881 | Bring your own LLM for OLS |
| OLS-1882 | Localization support for OLS messages |
| OLS-2154 | Manage MCP servers with MCP gateway |
| OLS-2491 | MCP client improvements |
| OLS-1884 | Customer policies to make output compliant |
| OLS-1886 | User ability to create support ticket from OLS console |
| OLS-1872 | BYOK -- internal web source integration |
| OLS-1679 | ROSA-aware answering in OpenShift Lightspeed |
| OLS-1806 | ARO-aware answering in OpenShift Lightspeed |
| OLS-1811 | Root cause analysis using OLS |
| OLS-1824 | Insights MCP adoption in OLS |
| OCPSTRAT-2985 | Make maxToolCallingIterations configurable via OLSConfig CR |
