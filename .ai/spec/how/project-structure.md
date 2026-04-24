# Project Structure -- Architecture

OpenShift LightSpeed (OLS) is a FastAPI service organized into four layers: `app/` (HTTP surface), `src/` (business logic), `utils/` (shared infrastructure), and `runners/` (process entry points). Each layer has strict import direction -- `app` depends on `src` and `utils`; `src` depends on `utils`; `runners` orchestrate startup.

## Module Map

### `ols/app/` -- FastAPI application layer

| Path | Purpose |
|---|---|
| `app/main.py` | Creates the `FastAPI` instance, registers middleware (metrics counter, security headers, request/response logging), calls `routers.include_routers(app)`, optionally mounts Gradio dev UI, and stores config status. The module-level `app` object is the ASGI entry point referenced by Uvicorn as `ols.app.main:app`. |
| `app/routers.py` | Single function `include_routers(app)` that attaches all endpoint routers. v1-prefixed: `ols`, `streaming_ols`, `mcp_client_headers`, `mcp_apps`, `tool_approvals`, `feedback`, `conversations`. Root-level: `health`, `metrics`, `authorized`. |
| `app/endpoints/ols.py` | `POST /v1/query` -- synchronous query endpoint. Orchestrates the full request lifecycle: auth, redaction, attachment processing, provider/model validation, quota check, `DocsSummarizer` invocation, conversation history storage, transcript storage, and quota consumption. |
| `app/endpoints/streaming_ols.py` | `POST /v1/streaming_query` -- streaming variant of the query endpoint. Uses the same `DocsSummarizer` but yields `StreamedChunk` objects via SSE. |
| `app/endpoints/health.py` | `/readiness` and `/liveness` probes. Readiness checks RAG index, LLM connectivity (with persistent caching of success), and cache backend. |
| `app/endpoints/feedback.py` | User feedback collection endpoint. |
| `app/endpoints/conversations.py` | Conversation history listing and deletion. |
| `app/endpoints/mcp_apps.py` | MCP application metadata endpoint. |
| `app/endpoints/mcp_client_headers.py` | MCP client header management endpoint. |
| `app/endpoints/tool_approvals.py` | Human-in-the-loop tool approval endpoint. |
| `app/endpoints/authorized.py` | Authorization check endpoint. |
| `app/metrics/metrics.py` | Prometheus metric definitions: `ols_rest_api_calls_total`, `ols_response_duration_seconds`, `ols_llm_calls_total`, `ols_llm_calls_failures_total`, `ols_llm_token_sent_total`, `ols_llm_token_received_total`, `ols_provider_model_configuration`. Exposes `GET /metrics` with auth. |
| `app/metrics/token_counter.py` | `GenericTokenCounter` (LangChain callback) and `TokenMetricUpdater` (context manager) for tracking per-request token usage and updating Prometheus counters. |
| `app/models/config.py` | All Pydantic configuration models: `Config`, `OLSConfig`, `LLMProviders`, `ProviderConfig`, `ModelConfig`, `DevConfig`, `ConversationCacheConfig`, `QuotaHandlersConfig`, `MCPServers`, `MCPServerConfig`, `ToolsApprovalConfig`, etc. |
| `app/models/models.py` | Request/response Pydantic models: `LLMRequest`, `LLMResponse`, `CacheEntry`, `SummarizerResponse`, `StreamedChunk`, `RagChunk`, `Attachment`, `TokenCounter`, health response models, etc. |

### `ols/src/` -- Core business logic

| Path | Purpose |
|---|---|
| `src/auth/auth.py` | `get_auth_dependency(ols_config, virtual_path)` factory that returns the configured `AuthDependencyInterface` implementation based on the `authentication_config.module` setting (`k8s`, `noop`, or `noop-with-token`). |
| `src/auth/auth_dependency_interface.py` | Abstract base class `AuthDependencyInterface(ABC)` -- all auth implementations must satisfy `async __call__(request) -> tuple[str, str, bool, str]` returning (user_id, user_name, skip_user_id_check, user_token). |
| `src/auth/k8s.py` | Kubernetes TokenReview-based authentication. Contains `K8sClientSingleton` for cluster ID and version retrieval. |
| `src/auth/noop.py` | No-op auth for local development (uses default user identity). |
| `src/auth/noop_with_token.py` | No-op auth that still extracts a token from the request. |
| `src/cache/cache.py` | Abstract `Cache(ABC)` base class. Defines compound key scheme (`user_id:conversation_id`), and abstract methods: `get`, `insert_or_append`, `delete`, `list`, `set_topic_summary`, `ready`. |
| `src/cache/cache_factory.py` | `CacheFactory.conversation_cache(config)` -- factory method that returns `InMemoryCache` or `PostgresCache` based on `config.type`. |
| `src/cache/in_memory_cache.py` | In-memory cache implementation with configurable max entries. |
| `src/cache/postgres_cache.py` | PostgreSQL-backed cache using SQLAlchemy/psycopg2. |
| `src/cache/cache_error.py` | `CacheError` exception class. |
| `src/llms/llm_loader.py` | `load_llm(provider, model, generic_llm_params)` -- resolves provider config, looks up the provider class in `LLMProvidersRegistry`, instantiates it, and calls `.load()` to return a LangChain LLM. Also defines `resolve_provider_config()` and error classes (`LLMConfigurationError`, `UnknownProviderError`, `UnsupportedProviderError`, `ModelConfigMissingError`). |
| `src/llms/providers/registry.py` | `LLMProvidersRegistry` -- class-level dict mapping provider type strings to `LLMProvider` subclasses. `@register_llm_provider_as("type")` decorator for self-registration. |
| `src/llms/providers/provider.py` | `LLMProvider(AbstractLLMProvider)` base class. Handles generic-to-provider parameter remapping, parameter validation against allowed parameter sets, dev config overrides, and HTTPX client construction (TLS, proxy, custom certificate stores). |
| `src/llms/providers/openai.py` | OpenAI provider implementation. |
| `src/llms/providers/azure_openai.py` | Azure OpenAI provider implementation. |
| `src/llms/providers/watsonx.py` | IBM WatsonX provider implementation. |
| `src/llms/providers/rhoai_vllm.py` | Red Hat OpenShift AI vLLM provider. |
| `src/llms/providers/rhelai_vllm.py` | RHEL AI vLLM provider. |
| `src/llms/providers/google_vertex.py` | Google Vertex AI provider. |
| `src/llms/providers/google_vertex_anthropic.py` | Google Vertex AI (Anthropic models) provider. |
| `src/llms/providers/fake_provider.py` | Fake provider for testing and load testing. |
| `src/prompts/prompts.py` | System prompt templates (`QUERY_SYSTEM_INSTRUCTION`, `TROUBLESHOOTING_SYSTEM_INSTRUCTION`). |
| `src/prompts/prompt_generator.py` | `GeneratePrompt` class that assembles `ChatPromptTemplate` from query, RAG context, history, system prompt, tool-calling flag, mode, cluster version, and optional skill content. |
| `src/query_helpers/query_helper.py` | `QueryHelper` base class for all query processing. Resolves provider/model defaults, selects system prompt by mode (`ask` or `troubleshooting`), and stores the LLM loader callable. |
| `src/query_helpers/docs_summarizer.py` | `DocsSummarizer(QueryHelper)` -- the central orchestrator. Prepares LLM, builds prompts with RAG context, manages the multi-round tool-calling loop via `iterate_with_tools()`, handles streaming and synchronous response modes. Contains `TokenBudgetTracker` integration for token accounting across prompt, RAG, history, tool definitions, tool results, and AI rounds. |
| `src/query_helpers/history_support.py` | `prepare_history()` -- retrieves and truncates conversation history from cache. |
| `src/query_helpers/attachment_appender.py` | `append_attachments_to_query()` -- serializes attachments into the query text. |
| `src/quota/quota_limiter.py` | Abstract `QuotaLimiter(ABC)` -- interface for `available_quota`, `ensure_available_quota`, `consume_tokens`, `revoke_quota`, `increase_quota`. |
| `src/quota/quota_limiter_factory.py` | Factory that creates quota limiter instances from config. |
| `src/quota/user_quota_limiter.py` | Per-user quota enforcement (PostgreSQL-backed). |
| `src/quota/cluster_quota_limiter.py` | Cluster-wide quota enforcement. |
| `src/quota/revokable_quota_limiter.py` | Quota limiter with periodic revocation support. |
| `src/quota/quota_exceed_error.py` | `QuotaExceedError` exception. |
| `src/quota/token_usage_history.py` | `TokenUsageHistory` -- records per-user token consumption to PostgreSQL for analytics. |
| `src/rag/hybrid_rag.py` | Hybrid RAG retrieval logic. |
| `src/rag_index/index_loader.py` | `IndexLoader` -- loads LlamaIndex vector indexes from configured reference content paths. Provides `get_retriever()` and `embed_model` for reuse. Excluded from MyPy type checking. |
| `src/skills/skills_rag.py` | `SkillsRAG` -- hybrid BM25 + vector retrieval for skill selection. `load_skills_from_directory()` parses skill files with YAML frontmatter. |
| `src/tools/tools.py` | `execute_tool_calls_stream()` -- runs resolved MCP tool calls with token budget enforcement and approval flow. `enforce_tool_token_budget()` truncates tool outputs that exceed remaining budget. |
| `src/tools/approval.py` | `PendingApprovalStoreBase` and `create_pending_approval_store()` -- human-in-the-loop tool approval infrastructure. |
| `src/tools/tools_rag/hybrid_tools_rag.py` | `ToolsRAG` -- hybrid BM25 + vector retrieval (using qdrant-client and rank-bm25) for filtering MCP tools by query relevance before sending to the LLM. |
| `src/ui/gradio_ui.py` | `GradioUI` -- optional development UI that mounts a Gradio interface onto the FastAPI app. |
| `src/config_status/config_status.py` | `extract_config_status()` and `store_config_status()` for telemetry about the active configuration. |

### `ols/utils/` -- Shared utilities

| Path | Purpose |
|---|---|
| `utils/config.py` | `AppConfig` singleton. Uses `__new__` for singleton enforcement. Lazy-initializes subsystems via `@property` and `@cached_property`: `conversation_cache`, `quota_limiters`, `token_usage_history`, `query_redactor`, `rag_index`, `rag_index_loader`, `tools_rag`, `skills_rag`, `pending_approval_store`. `reload_from_yaml_file()` parses YAML via `Config` Pydantic model and resets cached properties. The module-level `config` instance is the global singleton imported throughout the codebase. |
| `utils/logging_configurator.py` | `configure_logging()` -- sets up Python logging from config. |
| `utils/certificates.py` | `generate_certificates_file()` -- merges certifi CA bundle with any explicitly configured certificates into a single PEM file at `/tmp/ols.pem`. |
| `utils/ssl.py` | `get_ssl_version()` and `get_ciphers()` -- resolves TLS security profile settings for Uvicorn. |
| `utils/tls.py` | TLS utility functions for provider-level HTTPX clients: `ciphers_as_string()`, `min_tls_version()`, `ssl_tls_version()`. |
| `utils/redactor.py` | `Redactor` -- applies configured query filters (regex-based PII redaction) to queries and attachments before logging or LLM submission. |
| `utils/token_handler.py` | `TokenHandler` -- tiktoken-based token counting and RAG context truncation. `TokenBudgetTracker` -- per-request token budget management across categories (prompt, RAG, history, skill, tool definitions, tool results, AI rounds). `PromptTooLongError` exception. |
| `utils/mcp_utils.py` | `build_mcp_config()` and `get_mcp_tools()` -- resolves MCP server configurations, applies tool filtering via `ToolsRAG`, and fetches tools from MCP servers using `langchain-mcp-adapters`. |
| `utils/suid.py` | UUID generation and validation for conversation/user IDs. |
| `utils/environments.py` | `configure_gradio_ui_envs()` and `configure_hugging_face_envs()` -- sets environment variables before other imports. |
| `utils/checks.py` | `InvalidConfigurationError` and validation helpers. |
| `utils/errors_parsing.py` | `parse_generic_llm_error()` and `handle_known_errors()` -- translates LLM provider exceptions into HTTP status codes and user-facing messages. |
| `utils/postgres.py` | PostgreSQL connection utilities. |
| `utils/pyroscope.py` | Optional Pyroscope profiling integration. |

### `ols/runners/` -- Process entry points

| Path | Purpose |
|---|---|
| `runners/uvicorn.py` | `start_uvicorn(config)` -- configures and starts Uvicorn with `workers=1`, TLS settings from config, host/port selection, and log level. Entry point string is `"ols.app.main:app"`. |
| `runners/quota_scheduler.py` | `start_quota_scheduler(config)` -- spawns a daemon thread that periodically revokes/resets expired quotas in PostgreSQL via `INCREASE_QUOTA_STATEMENT` and `RESET_QUOTA_STATEMENT`. |

### `ols/plugins/` -- Plugin system

`plugins/__init__.py` provides `_import_modules_from_dir()` which dynamically imports all `.py` files from a given directory. Currently a placeholder; providers and tools are not yet migrated to this system.

### `ols/customize/` -- Customization hooks

`customize/ols/` is an empty directory (only `__pycache__`). Reserved for operator-injected customization files.

### Key standalone files

| Path | Purpose |
|---|---|
| `ols/constants.py` | All global constants: provider type strings (`PROVIDER_OPENAI`, etc.), `SUPPORTED_PROVIDER_TYPES`, `ModelFamily` enum, `QueryMode` enum (`ask`, `troubleshooting`), `GenericLLMParameters`, token budget constants (`DEFAULT_CONTEXT_WINDOW_SIZE`, `DEFAULT_MAX_TOKENS_FOR_RESPONSE`, `DEFAULT_TOOL_BUDGET_RATIO`), RAG constants (`RAG_CONTENT_LIMIT`, `RAG_SIMILARITY_CUTOFF`), cache constants, default auth module, HTTP headers to redact, attachment type/content type sets, SSL defaults, `RUNNING_IN_CLUSTER` flag. |
| `ols/version.py` | `__version__ = "1.0.12"` -- single source of truth for the service version. Used by `pyproject.toml` via `hatch`. |
| `runner.py` (project root) | Main entry point (`__main__`). Orchestrates the full startup sequence (see Data Flow below). |

## Data Flow

### Startup sequence (`runner.py`)

1. **Environment setup**: `configure_gradio_ui_envs()` sets Gradio-related environment variables. This must happen before importing `ols.config` because the config import triggers module-level code.

2. **Config import**: `from ols import config` -- this instantiates the `AppConfig` singleton at module level (`config: AppConfig = AppConfig()` in `ols/utils/config.py`). At this point, the config is empty.

3. **Config loading**: `config.reload_from_yaml_file(cfg_file)` reads the YAML file specified by `OLS_CONFIG_FILE` (defaulting to `olsconfig.yaml`), parses it into a `Config` Pydantic model via `yaml.safe_load()`, and runs `config.validate_yaml()`. Resets all cached properties (`_query_filters`, `_rag_index_loader`, `tools_rag`, `skills_rag`, etc.).

4. **Logging**: `configure_logging(config.ols_config.logging_config)` sets up Python logging.

5. **HuggingFace environment**: `configure_hugging_face_envs(config.ols_config)` sets `HF_HOME`, `TRANSFORMERS_CACHE`, etc.

6. **Certificate generation**: `generate_certificates_file()` merges certifi CA certs with any explicitly configured certificates into `/tmp/ols.pem`.

7. **K8s auth init** (conditional): If `use_k8s_auth()` is true, initializes `K8sClientSingleton` and retrieves the cluster ID. Fails fast if cluster ID is unavailable.

8. **Query redactor init**: Accessing `config.query_redactor` triggers lazy initialization of the `Redactor` instance from configured query filters.

9. **Pyroscope** (conditional): If `dev_config.pyroscope_url` is set, starts profiling.

10. **RAG index loading** (background thread): `threading.Thread(target=load_index)` starts a thread that accesses `config.rag_index`, which triggers `IndexLoader` initialization. This runs in parallel with server startup so the service can begin accepting health checks before indexes are loaded.

11. **Quota scheduler** (background thread): `start_quota_scheduler(config)` spawns a daemon thread for periodic quota revocation (only if PostgreSQL-backed quotas are configured).

12. **Uvicorn start**: `start_uvicorn(config)` starts the ASGI server with `workers=1`.

### Module-level initialization in `app/main.py`

When Uvicorn imports `ols.app.main`, additional module-level initialization occurs:

1. **Gradio UI** (conditional): If `config.dev_config.enable_dev_ui` is true, imports `GradioUI` and mounts it onto the FastAPI app. This import pulls in heavy dependencies (Matplotlib, Pillow, etc.) that are intentionally deferred.

2. **Metrics setup**: `metrics.setup_model_metrics(config)` initializes Prometheus gauges for all configured provider/model combinations.

3. **Router registration**: `routers.include_routers(app)` attaches all endpoint routers.

4. **Route path collection**: Builds `app_routes_paths` list for the metrics middleware to filter on.

5. **Config status** (conditional): If `user_data_collection.config_status_enabled` is true, extracts and stores config status for telemetry.

### Request flow (`POST /v1/query`)

```
Client
  |
  v
FastAPI middleware (log_requests_responses -> rest_api_counter)
  |
  v
ols.py: conversation_request()
  |-- process_request()
  |     |-- auth_dependency(request) -> (user_id, user_name, skip_check, token)
  |     |-- retrieve_conversation_id() -> new or existing UUID
  |     |-- redact_query() via config.query_redactor
  |     |-- redact_attachments()
  |     |-- append_attachments_to_query()
  |     |-- validate_requested_provider_model() via llm_loader.resolve_provider_config()
  |     |-- check_tokens_available() via quota_limiters
  |
  |-- generate_response()
  |     |-- DocsSummarizer(provider, model, system_prompt, mode, user_token, client_headers)
  |     |     |-- QueryHelper.__init__() -> resolves defaults, selects system prompt by mode
  |     |     |-- _prepare_llm() -> loads LLM via llm_loader.load_llm()
  |     |     |-- build_mcp_config() -> resolves MCP server configs
  |     |     |-- TokenBudgetTracker() -> initializes per-request token accounting
  |     |
  |     |-- .create_response(query, rag_retriever, user_id, conversation_id)
  |           |-- _prepare_prompt_context() -> RAG retrieval + truncation
  |           |-- skills_rag.retrieve_skill() -> optional skill injection
  |           |-- prepare_history() -> cache.get() + truncation
  |           |-- _build_final_prompt() -> GeneratePrompt().generate_prompt()
  |           |-- get_mcp_tools() -> tool filtering via ToolsRAG, tool fetching from MCP servers
  |           |-- iterate_with_tools() -> multi-round LLM + tool execution loop
  |                 |-- _invoke_llm() -> chain.astream() with optional bind_tools()
  |                 |-- _process_tool_calls_for_round() -> execute_tool_calls_stream()
  |
  |-- store_conversation_history() via config.conversation_cache
  |-- store_transcript() -> filesystem JSON
  |-- consume_tokens() via quota_limiters + token_usage_history
  |
  v
LLMResponse (conversation_id, response, referenced_documents, token counts, tool info)
```

## Key Abstractions

### AppConfig singleton (`ols/utils/config.py`)

The `AppConfig` class uses `__new__` to enforce a single instance. Subsystems are initialized lazily via `@property` (with manual `None`-check caching) or `@cached_property`. The module-level `config` variable is imported as `from ols import config` throughout the codebase, allowing `config.ols_config`, `config.conversation_cache`, `config.rag_index`, etc.

Lazy properties with manual caching (resettable on reload): `conversation_cache`, `quota_limiters`, `token_usage_history`, `query_redactor`, `rag_index_loader`.

Lazy properties with `@cached_property` (cleared by deleting from `__dict__` on reload): `mcp_servers_dict`, `tools_rag`, `skills_rag`.

### LLM Provider Registry (`ols/src/llms/providers/registry.py`)

`LLMProvidersRegistry` is a class with a `ClassVar[dict]` mapping provider type strings to `LLMProvider` subclasses. Providers self-register using the `@register_llm_provider_as("type")` decorator. `llm_loader.load_llm()` looks up the registry, instantiates the provider, and calls `.load()` to get a LangChain `BaseChatModel` or `LLM`.

The `LLMProvider` base class handles: generic-to-provider parameter remapping (e.g., `max_tokens_for_response` to `max_completion_tokens` for OpenAI), parameter validation against allowed sets, dev config overrides, and HTTPX client construction with TLS/proxy support.

### Auth dependency injection (`ols/src/auth/`)

`AuthDependencyInterface(ABC)` defines the contract: `async __call__(request) -> (user_id, user_name, skip_user_id_check, user_token)`. The `get_auth_dependency()` factory selects the implementation based on `authentication_config.module`. Endpoints use FastAPI's `Depends(auth_dependency)` pattern.

### Cache abstraction (`ols/src/cache/`)

`Cache(ABC)` defines the interface with compound keys (`user_id:conversation_id`). `CacheFactory` selects `InMemoryCache` or `PostgresCache` based on config. Both implementations store lists of `CacheEntry` objects (query/response pairs with metadata).

### Quota limiter abstraction (`ols/src/quota/`)

`QuotaLimiter(ABC)` defines `available_quota`, `ensure_available_quota`, `consume_tokens`. `QuotaLimiterFactory` creates instances. The quota scheduler runs in a separate daemon thread for periodic revocation/reset against PostgreSQL.

### Token budget tracking (`ols/utils/token_handler.py`)

`TokenBudgetTracker` manages the per-request token budget across categories: `PROMPT`, `RAG`, `HISTORY`, `SKILL`, `TOOL_DEFINITIONS`, `TOOL_RESULT`, `AI_ROUND`. It partitions the context window into prompt budget and tool budget based on `DEFAULT_TOOL_BUDGET_RATIO`, and enforces per-round caps via `tools_round_budget`.

### Router-based endpoint organization

All endpoints use `APIRouter` instances with tag grouping. The `include_routers()` function in `app/routers.py` attaches them with appropriate prefixes (`/v1` for business endpoints, none for health/metrics).

### Middleware stack (`app/main.py`)

Two middleware functions registered via `@app.middleware("")`:
- `log_requests_responses` -- debug-level request/response body and header logging (inner, runs first).
- `rest_api_counter` -- Prometheus histogram for response duration, counter for API calls, and security header injection on non-health endpoints (outer, runs second).

## Integration Points

### app -> src

- `app/endpoints/ols.py` imports `DocsSummarizer` from `src/query_helpers/docs_summarizer`
- `app/endpoints/ols.py` imports `get_auth_dependency` from `src/auth/auth`
- `app/endpoints/ols.py` imports `resolve_provider_config` from `src/llms/llm_loader`
- `app/endpoints/health.py` imports `load_llm` from `src/llms/llm_loader`
- `app/metrics/metrics.py` imports `get_auth_dependency` from `src/auth/auth`

### app -> utils

- `app/endpoints/ols.py` imports `errors_parsing`, `suid` from `utils/`
- All endpoint modules import `config` from `ols` (which is `ols/utils/config.py`)
- `app/main.py` imports `config`, `constants`, `version`

### src -> src (internal)

- `src/query_helpers/docs_summarizer.py` imports from `src/prompts/`, `src/tools/`, `src/auth/k8s`
- `src/query_helpers/query_helper.py` imports `load_llm` from `src/llms/llm_loader`
- `src/llms/llm_loader.py` imports `LLMProvidersRegistry` from `src/llms/providers/registry`

### src -> utils

- `src/query_helpers/docs_summarizer.py` imports `TokenHandler`, `TokenBudgetTracker` from `utils/token_handler`
- `src/query_helpers/docs_summarizer.py` imports `mcp_utils` from `utils/`
- `src/llms/providers/provider.py` imports `tls` from `utils/`

### utils/config.py -> src (lazy imports)

The `AppConfig` singleton imports factory classes from `src/` at module level: `CacheFactory`, `QuotaLimiterFactory`, `TokenUsageHistory`, `IndexLoader`, `SkillsRAG`, `ToolsRAG`. These are used in lazy `@property` methods, so the actual construction is deferred.

### runners -> everything

`runner.py` imports from `ols/constants`, `ols/runners/`, `ols/src/auth/`, `ols/utils/`, and `ols` (config). It orchestrates the startup sequence by calling into each subsystem.

## Implementation Notes

### Import order matters for lazy loading

`runner.py` calls `configure_gradio_ui_envs()` before `from ols import config`. This is required because importing `ols.config` triggers `ols/utils/config.py` module-level code, and Gradio environment variables must be set before any transitive Gradio imports. The comment in `runner.py` is explicit: "We import config here to avoid triggering import of anything else via our code before other envs are set (mainly the gradio)."

### Gradio dev UI is conditionally imported

In `app/main.py`, the `GradioUI` import is inside an `if config.dev_config.enable_dev_ui:` block. This avoids importing Matplotlib, Pillow, and other heavy Gradio dependencies in production. The Gradio UI mounts onto the existing FastAPI app instance and may replace it (the `app` variable is reassigned).

### Workers must be 1

`runners/uvicorn.py` passes `workers=config.ols_config.max_workers` to `uvicorn.run()`. However, the comment says "use workers=1 so config loaded can be accessed from other modules." The singleton `AppConfig` is process-local; multiple workers would each have independent config instances. The RAG index loading thread would also be per-worker.

### RAG index loads in background

The RAG index thread (`threading.Thread(target=load_index)`) starts before Uvicorn so the server can accept liveness probes immediately. The readiness probe at `/readiness` checks `index_is_ready()` which returns `False` until the index is loaded. The `IndexLoader` is excluded from MyPy type checking (`# type: ignore [attr-defined]`).

### Config reload resets cached properties differently

Properties using manual `None`-check caching (`_conversation_cache`, `_query_filters`, `_rag_index_loader`, `_tools_approval`, `_pending_approval_store`) are reset by setting the backing field to `None`. Properties using `@cached_property` (`mcp_servers_dict`, `tools_rag`, `skills_rag`) are reset by deleting the key from `self.__dict__`. Both patterns coexist in `reload_from_yaml_file()`.

### Auth dependency is resolved at module level

In `app/endpoints/ols.py` and `app/metrics/metrics.py`, `auth_dependency = get_auth_dependency(config.ols_config, virtual_path=...)` executes at import time. This means the auth module selection is fixed when the endpoint module is first imported and cannot change without restarting the process.

### Quota scheduler runs in a daemon thread

`start_quota_scheduler()` spawns a daemon thread with an infinite `while True` / `sleep(period)` loop. Because it is a daemon thread, it is killed when the main process exits. The scheduler directly manipulates PostgreSQL via psycopg2 (not through the quota limiter abstraction).

### Tool calling uses a multi-round streaming loop

`DocsSummarizer.iterate_with_tools()` implements the agentic tool-calling loop. It streams LLM chunks, detects tool call requests, executes them via MCP, appends results to the message history, and re-invokes the LLM -- up to `max_rounds` iterations. The final round binds tools with `tool_choice="none"` to force a text-only response. Tool outputs are truncated by `enforce_tool_token_budget()` when they exceed the remaining token budget.

### Entry point and build system

The ASGI entry point is `ols.app.main:app`. The build system uses `hatchling` with version sourced from `ols/version.py`. Dependencies are managed by `uv` (not pip/poetry). Python target is `>=3.12,<3.13`. The wheel package includes only the `ols` package.

### `RUNNING_IN_CLUSTER` detection

`ols/constants.py` sets `RUNNING_IN_CLUSTER` by checking for `KUBERNETES_SERVICE_HOST` and `KUBERNETES_SERVICE_PORT` environment variables at import time. This is a static, process-lifetime decision.

### Pending approval store uses deferred import

`AppConfig.pending_approval_store` uses an inline import (`from ols.src.tools.approval import create_pending_approval_store`) to avoid circular imports. This is one of the few places in the codebase where inline imports are used intentionally.
