# LLM Providers -- Architecture

The LLM provider subsystem translates a (provider name, model name) pair from configuration into a ready-to-use LangChain `BaseChatModel` (or `LLM`) instance, hiding backend differences behind a registry-and-base-class pattern.

## Module Map

### `ols/src/llms/llm_loader.py` -- Entry point

- `load_llm(provider, model, generic_llm_params)` -- The only function callers use. Reads `config.config.llm_providers`, resolves the provider config, looks up the provider class in the registry, instantiates it, and calls `.load()`.
- `resolve_provider_config(provider, model, providers_config)` -- Validates that the provider name exists in config and that the model is listed under that provider. Returns `ProviderConfig`.
- Exception hierarchy: `LLMConfigurationError` (base), `UnknownProviderError`, `UnsupportedProviderError`, `ModelConfigMissingError`.

### `ols/src/llms/providers/registry.py` -- Provider registry

- `LLMProvidersRegistry` -- Class-level dict (`llm_providers`) mapping string type names (e.g. `"openai"`, `"watsonx"`) to `LLMProvider` subclasses.
- `register_llm_provider_as(provider_type)` -- Decorator factory. Each provider file decorates its class with `@register_llm_provider_as(constants.PROVIDER_*)`, which triggers registration at import time.

### `ols/src/llms/providers/provider.py` -- Base class and parameter system

- `AbstractLLMProvider` -- ABC defining the `default_params` property and `load()` method.
- `LLMProvider(AbstractLLMProvider)` -- Concrete base. Constructor pipeline: `default_params` -> `_override_params` (merge caller params, then dev-config overrides) -> `_remap_to_llm_params` (generic-to-provider name translation) -> `_validate_parameters` (drop params not in the provider's allowed set).
- `_construct_httpx_client(use_custom_certificate_store, use_async)` -- Builds `httpx.Client` or `httpx.AsyncClient` with proxy, TLS security profile, and custom certificate store support. Used by OpenAI-compatible providers.
- `ProviderParameter(name, _type)` -- Frozen dataclass. The allowed-parameter sets use both name and type for validation (a parameter with the wrong type is rejected).
- Parameter sets: `AzureOpenAIParameters`, `OpenAIParameters`, `RHOAIVLLMParameters`, `RHELAIVLLMParameters`, `WatsonxParameters`, `FakeProviderParameters`, `GoogleVertexAnthropicParameters`, `GoogleVertexParameters`. Collected in `available_provider_parameters` dict keyed by provider type string.
- Generic-to-LLM mapping dicts: `AzureOpenAIParametersMapping`, `OpenAIParametersMapping`, `WatsonxParametersMapping`, etc. Collected in `generic_to_llm_parameters`.

### Provider implementations

| File | Class | Decorator key | LangChain class | Notes |
|---|---|---|---|---|
| `openai.py` | `OpenAI` | `"openai"` | `ChatOpenAI` | Reasoning params for o-series/gpt-5 models |
| `azure_openai.py` | `AzureOpenAI` | `"azure_openai"` | `AzureChatOpenAI` | Entra ID token caching; see below |
| `watsonx.py` | `Watsonx` | `"watsonx"` | `ChatWatsonx` | IBM-specific parameter names; see below |
| `rhoai_vllm.py` | `RHOAIVLLM` | `"rhoai_vllm"` | `ChatOpenAI` | OpenAI-compatible, no default URL |
| `rhelai_vllm.py` | `RHELAIVLLM` | `"rhelai_vllm"` | `ChatOpenAI` | OpenAI-compatible, no default URL |
| `google_vertex.py` | `GoogleVertex` | `"google_vertex"` | `ChatGoogleGenerativeAI` | Service account credentials from JSON |
| `google_vertex_anthropic.py` | `GoogleVertexAnthropic` | `"google_vertex_anthropic"` | `ChatAnthropicVertex` | Anthropic models on Vertex Model Garden |
| `fake_provider.py` | `FakeProvider` | `"fake_provider"` | `FakeListLLM` / `FakeStreamingListLLM` | Testing only; monkey-patches `bind_tools` |

### `ols/src/llms/providers/utils.py` -- Shared utilities

- `credentials_str_to_dict(credentials_json)` -- Parses a JSON string of service account credentials into a dict. Used by both Google Vertex providers.
- `VERTEX_AI_OAUTH_SCOPES` -- OAuth scope tuple for Vertex AI API access.

## Data Flow

```
Caller (QueryHelper / health check)
  |
  v
load_llm(provider="my_azure", model="gpt-4o", generic_llm_params={...})
  |
  |-- resolve_provider_config() validates provider + model exist in AppConfig
  |-- LLMProvidersRegistry.llm_providers[provider_config.type] gets the class
  |
  v
ProviderClass.__init__(model, provider_config, generic_llm_params)
  |-- default_params property builds provider-specific defaults
  |-- _override_params() merges: defaults < caller params < dev_config overrides
  |-- _remap_to_llm_params() translates generic names to provider-specific names
  |-- _validate_parameters() filters to only allowed (name, type) pairs
  |
  v
ProviderClass.load()
  |-- Constructs and returns a LangChain BaseChatModel (e.g. AzureChatOpenAI(**self.params))
  |
  v
Caller receives BaseChatModel, uses it in LangChain chains or agent loops
```

## Key Abstractions

### Registry: decorator-based auto-discovery

Each provider file decorates its class at module level:

```python
@register_llm_provider_as(constants.PROVIDER_OPENAI)
class OpenAI(LLMProvider):
    ...
```

Registration happens at import time. The `__init__.py` for the providers package is empty -- providers are imported implicitly because `provider.py` is imported by `registry.py`, and each provider file imports from `registry.py`. The loader file imports `LLMProvidersRegistry` from `registry.py`, which transitively triggers all provider registrations through Python's module system.

### Provider base class contract

Every provider must:
1. Define a `default_params` property returning a dict of provider-specific defaults.
2. Implement `load()` returning a `BaseChatModel` or `LLM`.

The base class handles parameter merging, remapping, and validation automatically in `__init__`.

### Parameter mapping: generic to provider-specific

Callers pass generic parameter names defined in `GenericLLMParameters`:

| Generic name | OpenAI / Azure / VLLM | WatsonX | Google Vertex |
|---|---|---|---|
| `max_tokens_for_response` | `max_completion_tokens` | `max_new_tokens` | `max_output_tokens` |
| `min_tokens_for_response` | (not mapped) | `min_new_tokens` | (not mapped) |
| `top_k` | (not mapped) | `top_k` | (not mapped) |
| `top_p` | (not mapped) | `top_p` | (not mapped) |
| `temperature` | (not mapped) | `temperature` | (not mapped) |

When a generic name has no mapping entry for a provider, it passes through unchanged and is then accepted or rejected by `_validate_parameters`.

### ModelFamily enum

`ModelFamily` in `ols/constants.py` defines `GPT` and `GRANITE`. It is not used by the provider subsystem itself but by downstream consumers (prompt generation, streaming chunk filtering) that detect model family by substring match against the model name: `ModelFamily.GRANITE in model_name`. Granite models get different agent instructions and different streaming-chunk skip logic.

### Provider-specific configuration precedence

Each provider follows the same pattern: read generic fields from `ProviderConfig` (url, credentials), then check for a provider-specific config object (e.g. `azure_config`, `watsonx_config`, `openai_config`) that overrides those values. The specific config always wins.

## Integration Points

### Callers

- **`QueryHelper`** (`ols/src/query_helpers/query_helper.py`) -- Base class for `DocsSummarizer` and other query helpers. Holds `llm_loader` as an injectable callable (defaults to `load_llm`). Called for main query answering, history compression, and tool execution flows.
- **Health endpoint** (`ols/app/endpoints/health.py`) -- Calls `load_llm` with the default provider/model to verify LLM connectivity at startup and during readiness probes.

### Configuration

- `ProviderConfig` and `ModelConfig` in `ols/app/models/config.py` supply all provider settings. `LLMProviders` holds the dict of named providers. The provider `type` field (defaulting to the provider name) determines which registered class handles it.
- `config.dev_config.llm_params` provides developer overrides that take highest precedence in `_override_params`.

### Constants

- Provider type strings: `PROVIDER_OPENAI`, `PROVIDER_AZURE_OPENAI`, `PROVIDER_WATSONX`, `PROVIDER_RHOAI_VLLM`, `PROVIDER_RHELAI_VLLM`, `PROVIDER_FAKE`, `PROVIDER_GOOGLE_VERTEX`, `PROVIDER_GOOGLE_VERTEX_ANTHROPIC` in `ols/constants.py`.
- `SUPPORTED_PROVIDER_TYPES` frozenset is checked during config validation (not by the registry).

### TLS and proxy

`LLMProvider._construct_httpx_client` reads `config.ols_config.proxy_config` for proxy URL, CA cert, and no-proxy host list, and `provider_config.tls_security_profile` for cipher and TLS version constraints. This affects all OpenAI-compatible providers (OpenAI, Azure OpenAI, RHOAI VLLM, RHELAI VLLM).

## Implementation Notes

### Adding a new provider

1. Create `ols/src/llms/providers/my_provider.py`.
2. Decorate the class: `@register_llm_provider_as(constants.PROVIDER_MY_PROVIDER)`.
3. Subclass `LLMProvider`. Implement `default_params` (property) and `load()`.
4. Add a constant `PROVIDER_MY_PROVIDER` to `ols/constants.py` and include it in `SUPPORTED_PROVIDER_TYPES`.
5. Define the parameter set and generic mapping in `provider.py` and add entries to `available_provider_parameters` and `generic_to_llm_parameters`.
6. Add the provider-specific config class to `ols/app/models/config.py` and wire it into `ProviderConfig`.

### Azure Entra ID token caching

`azure_openai.py` uses a module-level `TOKEN_CACHE` singleton (`TokenCache` dataclass). When credentials (API key) are not set, it fetches an Entra ID token via `ClientSecretCredential.get_token()` and caches it. The cache applies a 30-second leeway (`TOKEN_EXPIRATION_LEEWAY`) before the actual expiry to avoid using nearly-expired tokens. The cache is per-process (one per Uvicorn worker). The cache key is implicit -- there is a single `TOKEN_CACHE` instance, so it assumes one Azure provider per process.

### WatsonX parameter name translation

WatsonX uses IBM's `GenTextParamsMetaNames` constants for parameter keys instead of plain strings. The mapping in `WatsonxParametersMapping`:

| Generic | WatsonX (`GenParams.*`) |
|---|---|
| `min_tokens_for_response` | `MIN_NEW_TOKENS` |
| `max_tokens_for_response` | `MAX_NEW_TOKENS` |
| `top_k` | `TOP_K` |
| `top_p` | `TOP_P` |
| `temperature` | `TEMPERATURE` |

WatsonX also passes `params=self.params` to `ChatWatsonx` (as a nested dict), unlike OpenAI-compatible providers that spread params as `**self.params`.

### OpenAI reasoning model handling

Both `openai.py` and `azure_openai.py` detect o-series and gpt-5 models by name pattern (`self.model.startswith("o")` or `"gpt-5" in self.model`). For these models, temperature/top_p/frequency_penalty are omitted and reasoning parameters (effort, summary) are set instead. The `ModelParameters` config class provides `reasoning_effort`, `reasoning_summary`, and `verbosity` fields.

### HTTP client setup

`_construct_httpx_client` in the base class handles three concerns:
1. **Proxy**: reads `proxy_config.proxy_url`, creates `httpx.Proxy` with optional SSL context for HTTPS proxies, and sets up `no_proxy` host bypass via httpx mounts.
2. **Custom certificate store**: when `use_custom_certificate_store=True`, loads CA certs from `provider_config.certificates_store` (set automatically for OpenAI-compatible providers during config init).
3. **TLS security profile**: when `tls_security_profile` is configured, constrains minimum TLS version and cipher suites via `ssl.SSLContext`.

Each OpenAI-compatible provider creates both sync and async clients (`http_client` and `http_async_client`).

### Parameter validation by name and type

`_validate_parameters` checks both the parameter name and its Python type against the allowed set. A parameter named `temperature` with type `int` instead of `float` will be silently dropped with a warning log. This catches misconfigured parameters early but can be surprising if types do not match exactly.

### Provider type defaults to provider name

In `ProviderConfig.set_provider_type`, if no explicit `type` is set in the YAML, the provider's `name` is used as its type (lowercased). This means a provider named `"openai"` automatically gets type `"openai"` and resolves to the OpenAI provider class.
