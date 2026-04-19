# Configuration System

The configuration system loads, validates, and manages the single YAML file that drives all service behavior. It provides two-phase validation, credential security, and runtime reload without service restart.

## Behavioral Rules

### Loading

1. The service must read its configuration from a single YAML file whose path is specified by the `OLS_CONFIG_FILE` environment variable.
2. The YAML file must be parsed using `yaml.safe_load` -- no arbitrary Python object deserialization.
3. The configuration must be held in a process-wide singleton (`AppConfig`). All subsystems read from this singleton rather than loading the file independently.
4. Both `ols_config` and `llm_providers` top-level sections are required. The service must refuse to start if either is missing.

### Two-Phase Validation

5. **Phase 1 -- Structural validation** occurs during initial parsing (Pydantic model construction and `__init__` methods). It enforces: type correctness, required field presence, value range constraints, enum membership, credential file reading, and sub-object construction. A failure in Phase 1 prevents the `Config` object from being created.
6. **Phase 2 -- Semantic cross-validation** occurs after the full `Config` object is constructed, via `Config.validate_yaml()`. It enforces: `default_provider` references an existing provider in `llm_providers`, `default_model` references an existing model within that provider, file and directory paths are accessible on disk, TLS certificates exist when TLS is enabled, regex patterns in query filters compile, proxy URL format is valid, MCP server authorization headers resolve, and authentication module is a supported value.
7. Both phases must pass before the configuration is considered valid. A failure at any point must raise `InvalidConfigurationError` with a descriptive message.

### Dynamic Reload

8. The service must support reloading configuration from the YAML file at runtime without process restart, via `AppConfig.reload_from_yaml_file()`.
9. On reload, the new configuration must pass full two-phase validation before replacing the active configuration. If validation fails, the active configuration must remain unchanged and the error must be raised.
10. On successful reload, **stateless subsystems** must be re-initialized from the new configuration: query filters, RAG index loader, tool approval config, pending approval store, MCP server dictionary, tools RAG, and skills RAG.
11. On successful reload, **stateful subsystems** must persist their connections and state across reloads: conversation cache, quota limiters, and token usage history. These are not reset.

### Credential Handling

12. All credentials (API tokens, passwords, client secrets) must be read from files referenced by path in the configuration. Plaintext credential values must never appear directly in the YAML.
13. Credential reading uses `checks.read_secret()`, which supports both file paths (reads the file content) and directory paths (reads a default-named file within the directory, e.g., `apitoken` for API tokens).
14. The TLS key password, if needed, must also be read from a file path (`tls_key_password_path`), not stored as plaintext.

### Default Values

15. When `authentication_config.module` is not specified, it must default to `k8s` (Kubernetes RBAC).
16. When `context_window_size` is not specified per model, it must default to 128000.
17. When `max_tokens_for_response` is not specified per model, it must default to 4096.
18. When `history_compression_enabled` is not specified, it must default to `true`.
19. Proxy URL and no-proxy hosts must fall back to the `https_proxy`/`HTTPS_PROXY` and `no_proxy` environment variables, respectively, when not specified in config.

## Configuration Surface

### Top-Level Sections

The YAML file has four top-level sections:

| Section | Required | Purpose | Detail Spec |
|---------|----------|---------|-------------|
| `llm_providers` | Yes | Provider and model definitions | see what/llm-providers.md |
| `ols_config` | Yes | All service behavior configuration | (fields enumerated below) |
| `mcp_servers` | No | MCP server definitions | see what/tools.md |
| `dev_config` | No | Developer-mode flags | (fields enumerated below) |

### `ols_config` Fields

| Field Path | Type | Default | Purpose | Detail Spec |
|------------|------|---------|---------|-------------|
| `ols_config.default_provider` | string | (required) | Selects the default LLM provider by name | -- |
| `ols_config.default_model` | string | (required) | Selects the default model within the default provider | -- |
| `ols_config.authentication_config` | object | module=k8s | Auth module selection and K8s API settings | see what/auth.md |
| `ols_config.conversation_cache` | object | (required) | Cache backend selection and settings | see what/conversation-history.md |
| `ols_config.tls_config` | object | -- | Service endpoint TLS certificate and key paths | see what/security.md |
| `ols_config.proxy_config` | object | from env | HTTPS proxy URL, CA cert, no-proxy hosts | -- |
| `ols_config.query_filters` | list | none | PII redaction patterns (name, regex, replacement) | see what/security.md |
| `ols_config.logging_config` | object | INFO/WARNING | Log levels per component | see what/observability.md |
| `ols_config.user_data_collection` | object | all disabled | Feedback and transcript collection settings | see what/observability.md |
| `ols_config.tool_filtering` | object | none | Tool RAG filtering parameters | see what/tools.md |
| `ols_config.tools_approval` | object | never | Tool approval strategy and timeout | see what/tools.md |
| `ols_config.skills` | object | none | Skills directory and matching config | see what/skills.md |
| `ols_config.quota_handlers` | object | none | Quota limiter storage, scheduler, and limiters | see what/quota.md |
| `ols_config.reference_content` | object | none | RAG index paths and embeddings model | see what/rag.md |
| `ols_config.system_prompt_path` | string | none | Path to file containing custom system prompt | -- |
| `ols_config.history_compression_enabled` | bool | true | Toggle conversation history compression | -- |
| `ols_config.max_iterations` | int | mode-dependent | Tool-calling loop iteration cap (ask=5, troubleshooting=15) | -- |
| `ols_config.tool_round_cap_fraction` | float | 0.6 | Max fraction of remaining tool token budget usable per round (0.3--0.8) | -- |
| `ols_config.max_workers` | int | 1 | Number of concurrent workers | -- |
| `ols_config.expire_llm_is_ready_persistent_state` | int | -1 | Expiration for LLM readiness cache (-1 = never) | -- |
| `ols_config.extra_ca` | list | [] | Additional CA certificate file paths | see what/security.md |
| `ols_config.certificate_directory` | string | /tmp | Directory for assembled certificate stores | see what/security.md |
| `ols_config.tlsSecurityProfile` | object | none | TLS security profile for service endpoints | see what/security.md |

### `dev_config` Fields

| Field Path | Type | Default | Purpose |
|------------|------|---------|---------|
| `dev_config.enable_dev_ui` | bool | false | Enable Gradio development UI |
| `dev_config.disable_auth` | bool | false | Disable all authentication checks |
| `dev_config.disable_tls` | bool | false | Disable TLS on service endpoints |
| `dev_config.enable_system_prompt_override` | bool | false | Allow API requests to override system prompt |
| `dev_config.k8s_auth_token` | string | none | Static Kubernetes auth token for testing |
| `dev_config.run_on_localhost` | bool | false | Bind to localhost instead of all interfaces |
| `dev_config.uvicorn_port_number` | int | none | Custom HTTP server port |
| `dev_config.pyroscope_url` | string | none | Continuous profiling service URL |
| `dev_config.llm_params` | dict | {} | Override LLM parameters for testing |

### `mcp_servers` Fields

| Field Path | Type | Default | Purpose |
|------------|------|---------|---------|
| `mcp_servers[].name` | string | (required) | Unique server name |
| `mcp_servers[].url` | string | (required) | Server endpoint URL |
| `mcp_servers[].timeout` | int | none | Request timeout in seconds |
| `mcp_servers[].headers` | dict | {} | Auth headers (file paths, `kubernetes` placeholder, or `client` placeholder) |

### `llm_providers` Fields

Each provider entry under `llm_providers` supports:

| Field Path | Type | Default | Purpose |
|------------|------|---------|---------|
| `llm_providers[].name` | string | (required) | Provider name |
| `llm_providers[].type` | string | =name | Provider type (openai, azure_openai, watsonx, rhoai_vllm, rhelai_vllm, google_vertex, google_vertex_anthropic, fake_provider) |
| `llm_providers[].url` | URL | none | Provider endpoint URL |
| `llm_providers[].credentials_path` | string | none | Path to credential file or directory |
| `llm_providers[].models[]` | list | (required, >= 1) | Model definitions |
| `llm_providers[].models[].name` | string | (required) | Model identifier |
| `llm_providers[].models[].context_window_size` | int | 128000 | Context window size in tokens |
| `llm_providers[].models[].parameters.max_tokens_for_response` | int | 4096 | Tokens reserved for response |
| `llm_providers[].models[].parameters.tool_budget_ratio` | float | 0.25 | Fraction of context window for tool outputs (0.1--0.6) |
| `llm_providers[].models[].parameters.reasoning_effort` | enum | low | Reasoning effort level (low, medium, high) |
| `llm_providers[].models[].parameters.reasoning_summary` | enum | concise | Reasoning summary style (auto, concise, detailed) |
| `llm_providers[].models[].parameters.verbosity` | enum | low | General verbosity level (low, medium, high) |
| `llm_providers[].models[].options` | dict | none | Arbitrary key-value model options |
| `llm_providers[].<type>_config` | object | none | Provider-specific config (at most one per provider) |
| `llm_providers[].tlsSecurityProfile` | object | none | TLS security profile for provider connection |

## Constraints

### Structural Invariants

1. The `Config` object must always have both `ols_config` and `llm_providers` populated. Missing either is a fatal startup error.
2. Every provider must have at least one model. A provider with zero models is rejected during Phase 1.
3. At most one provider-specific configuration block (e.g., `openai_config`, `azure_openai_config`) may appear per provider entry. Multiple blocks are rejected.
4. The provider-specific config block, if present, must match the provider's `type`. A mismatch is rejected.
5. MCP server names must be unique across all entries. Duplicates are rejected by Pydantic validation.
6. Per-model `context_window_size` must be strictly greater than `max_tokens_for_response + max_tokens_for_tools`. This is validated after tool budgets are computed.
7. `tool_budget_ratio` must be between 0.1 and 0.6. `tool_round_cap_fraction` must be between 0.3 and 0.8.

### Security Constraints

8. Credentials must never be stored as plaintext values in the YAML. They must always be file path references resolved at load time.
9. TLS cannot be enabled without both a certificate path and a key path configured.
10. The TLS security profile must enforce a minimum of TLS 1.2. The `OldType` profile is rejected.
11. MCP servers using the `kubernetes` authorization header placeholder require the authentication module to be `k8s` or `noop-with-token`. With any other auth module, the server is excluded (not a fatal error -- it is silently filtered out with a warning).

### Cross-Validation Constraints

12. `default_provider` must name a provider that exists in `llm_providers`.
13. `default_model` must name a model that exists within the default provider's model list.
14. `conversation_cache.type` must be either `memory` or `postgres`. The corresponding sub-section (`memory` or `postgres`) must be present when the type is specified.
15. `authentication_config.module` must be one of: `k8s`, `noop`, `noop-with-token`.
16. Provider `type` must be one of the supported provider types: openai, azure_openai, watsonx, rhoai_vllm, rhelai_vllm, google_vertex, google_vertex_anthropic, fake_provider.
17. `project_id` is required when provider type is `watsonx`.

### Tool Budget Computation

18. When MCP servers are configured, each model's `max_tokens_for_tools` is computed as `context_window_size * tool_budget_ratio`. When no MCP servers are configured, `max_tokens_for_tools` is 0.
19. After computing tool budgets, the system validates that `context_window_size > max_tokens_for_response + max_tokens_for_tools` for every model. Violation is a fatal configuration error.

## Planned Changes

- [PLANNED: OLS-2874] Enable Skills Configuration in OLSConfig CRD -- expose `ols_config.skills` through the operator's custom resource definition, allowing skills to be configured declaratively via the OLSConfig CR.
