# Configuration -- Architecture

The configuration subsystem loads, validates, and exposes the entire service configuration through a singleton `AppConfig` object. Every other module in OLS reads its settings from this singleton rather than parsing files or environment variables directly.

## Module Map

### `ols/app/models/config.py` (Pydantic model hierarchy)

The root model is `Config`, which composes the top-level sections:

```
Config (root)
  ├── llm_providers: LLMProviders
  │     └── providers: dict[str, ProviderConfig]
  │           ├── type, url, credentials, project_id
  │           ├── models: dict[str, ModelConfig]
  │           │     ├── name, context_window_size, max_tokens_for_tools
  │           │     └── parameters: ModelParameters (max_tokens_for_response, reasoning_*, tool_budget_ratio)
  │           ├── tls_security_profile: TLSSecurityProfile
  │           └── <provider>_config: OpenAIConfig | AzureOpenAIConfig | WatsonxConfig | ...
  ├── ols_config: OLSConfig
  │     ├── conversation_cache: ConversationCacheConfig → InMemoryCacheConfig | PostgresConfig
  │     ├── logging_config: LoggingConfig
  │     ├── reference_content: ReferenceContent → list[ReferenceContentIndex]
  │     ├── authentication_config: AuthenticationConfig
  │     ├── tls_config: TLSConfig
  │     ├── tls_security_profile: TLSSecurityProfile
  │     ├── proxy_config: ProxyConfig
  │     ├── quota_handlers: QuotaHandlersConfig → PostgresConfig, SchedulerConfig, LimitersConfig
  │     ├── user_data_collection: UserDataCollection
  │     ├── query_filters: list[QueryFilter]
  │     ├── default_provider, default_model, max_iterations, max_workers
  │     ├── tool_filtering: ToolFilteringConfig
  │     ├── tools_approval: ToolsApprovalConfig
  │     ├── skills: SkillsConfig
  │     └── tool_round_cap_fraction: float
  ├── dev_config: DevConfig
  │     └── enable_dev_ui, disable_auth, disable_tls, pyroscope_url, run_on_localhost, ...
  └── mcp_servers: MCPServers
        └── servers: list[MCPServerConfig]
              ├── name, url, timeout
              └── headers: dict[str, str] (raw) / _resolved_headers: dict[str, str] (resolved)
```

Provider-specific config classes (`OpenAIConfig`, `AzureOpenAIConfig`, `WatsonxConfig`, `RHOAIVLLMConfig`, `RHELAIVLLMConfig`, `GoogleVertexAnthropicConfig`, `GoogleVertexConfig`, `FakeConfig`) inherit from `ProviderSpecificConfig` (which carries `url`, `token`, `api_key`) and use `extra="forbid"` to reject typos.

### `ols/utils/config.py` (AppConfig singleton)

- `AppConfig` -- singleton class (`__new__` pattern) that wraps `Config` and provides lazy-initialized accessors for expensive resources.
- Module-level `config: AppConfig = AppConfig()` -- the singleton instance, re-exported via `ols/__init__.py` as `from ols import config`.
- `reload_from_yaml_file()` -- reads YAML, builds `Config`, runs `validate_yaml()`, replaces the internal `config` attribute, and clears cached resources.
- `_load_config_from_yaml_stream()` -- static method that handles YAML parsing and Pydantic construction in one step.

### `ols/utils/checks.py` (validation helpers)

- `InvalidConfigurationError` -- the single exception type for all configuration validation failures.
- `read_secret()` -- reads a secret value from a file path; if the path is a directory, appends a default filename (e.g., `apitoken`). Supports `raise_on_error=False` for optional secrets.
- `get_attribute_from_file()` -- reads an arbitrary attribute value from a file path stored in config.
- `is_valid_http_url()` -- validates URL scheme is http or https.
- `dir_check()` / `file_check()` -- assert paths exist and are readable.
- `get_log_level()` -- converts string log level names to `logging` integer constants.
- `resolve_headers()` -- resolves MCP server authorization headers: reads secret files, or preserves `"kubernetes"` / `"client"` placeholders for runtime substitution.
- `validate_mcp_servers()` -- filters MCP server list by resolving headers; drops servers whose headers cannot be resolved.

## Data Flow

### Startup path (runner.py)

```
1. runner.py reads OLS_CONFIG_FILE env var (default: olsconfig.yaml)
2. config.reload_from_yaml_file(cfg_file)
   a. Open YAML file, yaml.safe_load() → raw dict
   b. Config(data, ignore_llm_secrets, ignore_missing_certs)
      - Phase 1 (structural): Pydantic parsing + custom __init__ methods
        - Each model's __init__ manually extracts fields from raw dicts
        - model_validator(mode="before") runs on ModelConfig, ProxyConfig
        - model_validator(mode="after") runs on PostgresConfig, UserDataCollection, MCPServers
        - field_validator runs on ModelConfig.options, ModelParameters.tool_budget_ratio
      - Phase 2 (cross-field / semantic): Config.__init__ calls
        - _validate_mcp_servers() -- resolves headers with auth context
        - _compute_tool_budgets() -- sets max_tokens_for_tools per model based on MCP presence
   c. config.validate_yaml()
      - Phase 3 (file-system / external): walks the entire tree calling validate_yaml()
        on each sub-config, checking file existence, URL validity, TLS certs, etc.
3. config is now fully initialized; other modules access it via `from ols import config`
```

### Access path (runtime)

```
any module → from ols import config → config.ols_config / config.llm_config / config.dev_config
                                     → config.conversation_cache  (lazy, via @property)
                                     → config.rag_index            (lazy, via @property)
                                     → config.tools_rag            (lazy, via @cached_property)
                                     → config.skills_rag           (lazy, via @cached_property)
                                     → config.quota_limiters       (lazy, via @property)
                                     → config.query_redactor       (lazy, via @property)
```

### Reload path

```
config.reload_from_yaml_file(path)
  → build new Config from YAML (same as startup)
  → replace self.config
  → clear all lazy caches:
      - self._query_filters = None
      - self._rag_index_loader = None
      - self._tools_approval = None
      - self._pending_approval_store = None
      - del self.__dict__["mcp_servers_dict"]   (cached_property)
      - del self.__dict__["tools_rag"]          (cached_property)
      - del self.__dict__["skills_rag"]         (cached_property)
  → next access re-initializes each resource lazily
```

## Key Abstractions

### Two lazy-caching strategies

`AppConfig` uses two different patterns for lazy initialization, and they are invalidated differently on reload:

1. **`@property` with `_field is None` guard** -- used for `conversation_cache`, `quota_limiters`, `token_usage_history`, `query_redactor`, `rag_index_loader`, `tools_approval`, `pending_approval_store`. Cleared by setting `self._field = None` in `reload_from_yaml_file()`.

2. **`@cached_property`** -- used for `mcp_servers_dict`, `tools_rag`, `skills_rag`. These are stored in `self.__dict__` by Python's `cached_property` protocol. Cleared by `del self.__dict__["key"]` in `reload_from_yaml_file()`.

### Two-phase validation

1. **Structural validation** -- Pydantic model construction. Field types, `field_validator`, `model_validator(mode="before"|"after")`, and custom `__init__` logic that reads secrets from files and builds nested sub-configs.

2. **Semantic validation** -- explicit `validate_yaml()` methods called after construction. These check cross-cutting concerns that require file-system access (TLS cert existence, directory readability, regex compilation) or cross-model references (default_provider must exist in llm_providers, default_model must exist in that provider).

### Credential file resolution

`checks.read_secret(data, path_key, default_filename)` implements a two-mode lookup:
- If the path points to a **file**, reads it directly.
- If the path points to a **directory**, appends `default_filename` (e.g., `apitoken`) and reads that file.

This supports both Kubernetes secret volume mounts (directory of files) and explicit file paths. The `directory_name_expected` flag (used by Azure) enforces directory-only mode. The `raise_on_error` flag controls whether missing secrets are fatal or silently ignored.

### TLS profile to cipher suite mapping

`TLSSecurityProfile` validates against `ols/utils/tls.py`:
- Profile types: `OldType` (rejected at validation), `IntermediateType`, `ModernType`, `Custom`.
- Each profile maps to a predefined cipher suite list in `tls.TLS_CIPHERS`.
- Minimum TLS version is enforced: `VersionTLS12` is the floor. Anything below is rejected.
- For non-Custom profiles, any user-specified ciphers are validated against the profile's allowed set.
- `ols/utils/ssl.py` consumes the validated profile to configure Uvicorn's SSL context.

### Provider-specific config dispatch

`ProviderConfig.set_provider_specific_configuration()` uses a `match/case` on `self.type` to instantiate the correct typed config class. It enforces that at most one `<provider>_config` block is present and that it matches the declared provider type. Each provider-specific class uses `extra="forbid"` to catch misconfigured fields early.

## Integration Points

- **Every module** reads config through `from ols import config` (re-export of the singleton).
- **runner.py** -- loads config, then passes it to `start_uvicorn()`, `start_quota_scheduler()`, and triggers `load_index()` in a background thread.
- **FastAPI app (main.py)** -- reads `config.dev_config`, `config.ols_config` at module import time (top-level statements), so config must be loaded before the app module is imported.
- **LLM loader** -- reads `config.llm_config.providers[name]` and provider-specific sub-configs to construct LangChain LLM instances.
- **Cache factory** -- reads `config.ols_config.conversation_cache` to create in-memory or Postgres cache.
- **RAG index loader** -- reads `config.ols_config.reference_content` to load vector indexes.
- **Quota system** -- reads `config.ols_config.quota_handlers` for storage and limiter configuration.
- **MCP tools** -- reads `config.mcp_servers` for server URLs and resolved headers.
- **Tools/Skills RAG** -- `config.tools_rag` and `config.skills_rag` are created from `tool_filtering` and `skills` config sections respectively, sharing the embedding model resolved via `_resolve_embed_model()`.
- **TLS/SSL** -- `config.ols_config.tls_config` and `tls_security_profile` feed into Uvicorn's SSL parameters.

## Implementation Notes

### Config models use manual `__init__` instead of Pydantic's declarative parsing

Most config models override `__init__` to accept a raw `dict` (or `Optional[dict]`) and manually extract fields. This is a legacy pattern predating Pydantic v2's improved `model_validator`. It means that adding a new field requires wiring it in the `__init__` method, not just declaring it on the class. Some models (`DevConfig`, `LoggingConfig`, `PostgresConfig`, `UserDataCollection`, `ModelConfig`, `ModelParameters`) use Pydantic's standard `**kwargs` construction instead.

### Reload does not re-initialize the singleton

`reload_from_yaml_file()` replaces `self.config` and clears cached resources, but does **not** create a new `AppConfig` instance. All modules holding a reference to the singleton continue to see the updated config. The `_conversation_cache` and `_quota_limiters` are **not** cleared on reload -- they persist across reloads. Only `_query_filters`, `_rag_index_loader`, `_tools_approval`, `_pending_approval_store`, and the three `cached_property` values are cleared.

### Adding a new config section

1. Create a Pydantic `BaseModel` subclass in `ols/app/models/config.py`.
2. Add it as an `Optional` field on the appropriate parent model (`OLSConfig`, `Config`, etc.).
3. Wire it in the parent's `__init__` method (extract from `data` dict, instantiate).
4. If it needs file-system or cross-field checks, add a `validate_yaml()` method and call it from the parent's `validate_yaml()`.
5. If it needs lazy resource initialization, add a `@property` or `@cached_property` on `AppConfig` and clear it in `reload_from_yaml_file()`.

### `_compute_tool_budgets()` is called during `Config.__init__`

This cross-cutting validation sets `max_tokens_for_tools` on every `ModelConfig` based on whether MCP servers are configured. It also enforces that `max_tokens_for_response + max_tokens_for_tools < context_window_size`. This runs after MCP server validation, so the server list is already filtered.

### Proxy config falls through to environment variables

`ProxyConfig` reads `https_proxy` / `HTTPS_PROXY` and `no_proxy` environment variables as defaults via `Field(default_factory=...)`. Explicit YAML values override these. The `model_validator(mode="before")` also sets `proxy_url` from env if not provided in the input dict, creating a double-fallback path.

### MCP header resolution happens at config load time, not request time

Secret file contents are read into `_resolved_headers` during `Config.__init__` (via `_validate_mcp_servers()` -> `checks.validate_mcp_servers()` -> `checks.resolve_headers()`). The special placeholders `"kubernetes"` and `"client"` are preserved as-is and substituted at request time by the MCP client layer.
