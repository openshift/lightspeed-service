# LLM Provider System

The provider system loads, configures, and communicates with LLM backends so that the core query pipeline operates identically regardless of which backend is in use.

## Behavioral Rules

### Provider Contract

Every provider must satisfy all of the following:

1. The system must maintain a provider registry. Each provider implementation registers itself by provider type string at import time, so that adding a new provider never requires modifying core loading code.

2. When a provider/model combination is requested, the system must resolve the provider configuration from `olsconfig.yaml`, verify that the provider name exists and the model is listed under that provider, then look up the provider type in the registry to instantiate and load the LLM. If the provider name is not in configuration, a distinct "unknown provider" error must be raised. If the model is not configured for the provider, a distinct "model config missing" error must be raised. If the provider type has no registered implementation, a distinct "unsupported provider" error must be raised.

3. Each provider must define default parameters appropriate to its backend. These defaults are the baseline for every LLM call.

4. Parameters must follow a three-level precedence, applied in this order:
   - **Provider defaults** (lowest priority) -- hardcoded sensible values for the backend.
   - **Per-request parameters** -- caller-supplied generic parameters override defaults.
   - **Admin overrides** (highest priority) -- values from the developer/debug configuration override everything, including per-request parameters.

5. The system must translate generic parameter names to provider-specific names before passing them to the backend. Generic names include `max_tokens_for_response`, `min_tokens_for_response`, `temperature`, `top_k`, and `top_p`. Each provider declares its own mapping. Parameters with no mapping pass through with their original name.

6. After remapping, the system must validate parameters against the provider's declared set of accepted parameter names and types. Parameters not recognized by the target provider must be silently filtered out with a warning log, not cause errors. Parameters with a recognized name but a `None` value must be allowed through.

7. Each provider must produce both a synchronous and an asynchronous HTTP client (where applicable) and pass them to the LLM backend library. Both clients must respect the same proxy, TLS, and certificate settings.

8. Each provider must support tool binding: the LLM instance returned by `load()` must accept `bind_tools()` to attach tool/function definitions for tool-calling workflows.

9. Provider-specific configuration (e.g., `openai_config`, `azure_openai_config`) takes precedence over top-level provider fields (`url`, `credentials`) whenever both are present.

10. Credentials must be read from files on disk (via `credentials_path`), not from environment variables or inline config values. The file path points to a directory or file containing the secret.

### Reasoning Model Support

11. Models must be classified as reasoning-capable or standard based on their name. A model is reasoning-capable if its name contains `gpt-5` or starts with `o` (the OpenAI o-series pattern).

12. For reasoning-capable models, the provider must set reasoning-specific parameters (`reasoning_effort`, `reasoning_summary`, `verbosity`) drawn from the model's `parameters` configuration. Standard sampling parameters (`temperature`, `top_p`, `frequency_penalty`) must not be set.

13. For standard (non-reasoning) models, the provider must set sampling parameters with sensible defaults (`temperature`, `top_p`, `frequency_penalty`). Reasoning parameters must not be set.

14. `reasoning_effort` and `verbosity` accept values `low`, `medium`, or `high`. `reasoning_summary` accepts `auto`, `concise`, or `detailed`.

### HTTP Client Requirements

15. All providers communicating over HTTP must support proxy configuration. When `ols_config.proxy_config.proxy_url` is set, all HTTP clients must route through that proxy. If the proxy URL is HTTPS, a proxy-specific SSL context must be created using the configured proxy CA certificate.

16. Hosts listed in `ols_config.proxy_config.no_proxy_hosts` must bypass the proxy entirely.

17. Providers that use custom certificate stores (OpenAI, Azure OpenAI, RHOAI vLLM, RHELAI vLLM) must load the certificate bundle from `certificates_store` and use it for TLS verification.

18. When `tlsSecurityProfile` is configured on the provider, the HTTP client must enforce the specified minimum TLS version and cipher suite list. When no security profile is set, the client must use default TLS verification (or the custom certificate store if applicable).

## Per-Provider Deviations

The following sections describe only what differs from the standard contract above. Behavior not mentioned is identical to the contract.

### OpenAI (`openai`)

19. Default URL: `https://api.openai.com/v1`. Uses `ChatOpenAI` from LangChain. No deviations from the standard contract. Uses custom certificate store.

### Azure OpenAI (`azure_openai`)

20. Must support two authentication modes:
    - **API key**: When `credentials` (or `azure_openai_config.api_key`) is set, use it as the `api_key` parameter.
    - **Entra ID service principal**: When no API key is present, obtain an Azure AD token using `tenant_id`, `client_id`, and `client_secret` from `azure_openai_config`. The token is scoped to `https://cognitiveservices.azure.com/.default`.

21. Azure AD tokens must be cached per process. The cached token must be refreshed when it expires, with a 30-second safety margin before the actual expiration time. If token retrieval fails, the provider must log the error and return `None` for the token (degraded operation), not crash.

22. If Entra ID auth is selected but any of `tenant_id`, `client_id`, or `client_secret` is missing, the provider must raise a specific error naming the absent field.

23. `api_version` must be configurable, defaulting to `2024-02-15-preview`. `deployment_name` is configured separately from `model` name.

24. Uses `AzureChatOpenAI` from LangChain. Uses custom certificate store.

### WatsonX (`watsonx`)

25. Requires `project_id` to be set in configuration. If `project_id` is missing, the system must reject the configuration at startup (not at first request).

26. Requires `credentials` (API key) to be set. If missing, the provider must raise a ValueError at load time.

27. Generic parameter names map to WatsonX-specific names: `max_tokens_for_response` maps to `MAX_NEW_TOKENS`, `min_tokens_for_response` maps to `MIN_NEW_TOKENS`, `temperature` maps to WatsonX's `TEMPERATURE`, `top_k` to `TOP_K`, `top_p` to `TOP_P`. Additional WatsonX-specific parameters include `DECODING_METHOD`, `RANDOM_SEED`, and `REPETITION_PENALTY`.

28. Parameters are passed to the LLM constructor in a `params` dict (not as top-level keyword arguments). Uses `ChatWatsonx` from LangChain IBM. Does not use httpx clients or custom certificate stores.

29. Default URL: `https://us-south.ml.cloud.ibm.com`.

### RHOAI vLLM (`rhoai_vllm`)

30. Uses the OpenAI-compatible API via `ChatOpenAI`. No default URL is defined (falls back to `https://api.openai.com/v1` but the admin must configure the actual endpoint). Uses custom certificate store.

31. Sets standard sampling defaults (`temperature`, `top_p`, `frequency_penalty`) regardless of model name -- no reasoning model detection.

### RHELAI vLLM (`rhelai_vllm`)

32. Identical to RHOAI vLLM in behavior. Registered under a different type identifier (`rhelai_vllm`) to distinguish the deployment context in configuration.

### Google Vertex AI - Gemini (`google_vertex`)

33. Requires `credentials` as a JSON string containing a Google service account key. The JSON is parsed and used to create Google OAuth2 credentials scoped to `https://www.googleapis.com/auth/cloud-platform`.

34. `project` and `location` are configured via `google_vertex_config`. Default location: `global`. Uses `ChatGoogleGenerativeAI` from LangChain with `vertexai=True`.

35. Does not use httpx clients or custom certificate stores. `max_tokens_for_response` maps to `max_output_tokens`.

### Google Vertex AI - Anthropic/Claude (`google_vertex_anthropic`)

36. Same credential handling as Google Vertex (Gemini): requires JSON service account key, same OAuth2 scope.

37. `project` and `location` are configured via `google_vertex_anthropic_config`. Default location: `us-east5`. Uses `ChatAnthropicVertex` from LangChain.

38. Does not use httpx clients or custom certificate stores. `max_tokens_for_response` maps to `max_output_tokens`.

### Fake Provider (`fake_provider`)

39. Returns static preconfigured responses for testing. Supports a streaming mode that splits the response into chunks with a configurable sleep interval, and a non-streaming mode that returns the full response at once.

40. Accepts `bind_tools()` but ignores the tools (returns the same LLM instance unchanged).

41. Not intended for production use. Configured via `fake_provider_config` with fields: `stream`, `mcp_tool_call`, `response`, `chunks`, `sleep`.

## Configuration Surface

- `llm_providers[].name` -- Provider instance name (used as the lookup key).
- `llm_providers[].type` -- Provider type string (must match a registered type: `openai`, `azure_openai`, `watsonx`, `rhoai_vllm`, `rhelai_vllm`, `google_vertex`, `google_vertex_anthropic`, `fake_provider`). Defaults to the provider name if not specified.
- `llm_providers[].url` -- Base URL for the LLM API endpoint.
- `llm_providers[].credentials_path` -- Path to file or directory containing the API key/token.
- `llm_providers[].project_id` -- Required for WatsonX; project identifier.
- `llm_providers[].api_version` -- Azure OpenAI API version (default: `2024-02-15-preview`).
- `llm_providers[].deployment_name` -- Azure OpenAI deployment name.
- `llm_providers[].models[]` -- List of model configurations under this provider.
- `llm_providers[].models[].name` -- Model identifier passed to the backend.
- `llm_providers[].models[].context_window_size` -- Token context window (default: 128000).
- `llm_providers[].models[].credentials_path` -- Model-level credential override.
- `llm_providers[].models[].parameters.max_tokens_for_response` -- Max tokens reserved for the LLM response (default: 4096).
- `llm_providers[].models[].parameters.reasoning_effort` -- Reasoning effort level: `low`, `medium`, or `high` (default: `low`).
- `llm_providers[].models[].parameters.reasoning_summary` -- Reasoning summary mode: `auto`, `concise`, or `detailed` (default: `concise`).
- `llm_providers[].models[].parameters.verbosity` -- Verbosity for reasoning models: `low`, `medium`, or `high` (default: `low`).
- `llm_providers[].models[].parameters.tool_budget_ratio` -- Fraction of context window reserved for tool outputs (default: 0.25, range: 0.1--0.6).
- `llm_providers[].models[].options` -- Arbitrary key-value options dict passed through to the model.
- `llm_providers[].tlsSecurityProfile` -- TLS security profile with `type`, `minTLSVersion`, and `ciphers`.
- `llm_providers[].openai_config` -- Provider-specific: `url`, `credentials_path`.
- `llm_providers[].azure_openai_config` -- Provider-specific: `url`, `deployment_name`, `credentials_path` (directory containing `apitoken`, `client_id`, `tenant_id`, `client_secret` files).
- `llm_providers[].watsonx_config` -- Provider-specific: `url`, `credentials_path`, `project_id`.
- `llm_providers[].rhoai_vllm_config` -- Provider-specific: `url`, `credentials_path`.
- `llm_providers[].rhelai_vllm_config` -- Provider-specific: `url`, `credentials_path`.
- `llm_providers[].google_vertex_config` -- Provider-specific: `project`, `location`.
- `llm_providers[].google_vertex_anthropic_config` -- Provider-specific: `project`, `location`.
- `llm_providers[].fake_provider_config` -- Testing: `stream`, `mcp_tool_call`, `response`, `chunks`, `sleep`.
- `dev_config.llm_params` -- Admin/developer override parameters applied at highest precedence.
- `ols_config.proxy_config.proxy_url` -- HTTP/HTTPS proxy URL for LLM traffic.
- `ols_config.proxy_config.proxy_ca_cert_path` -- CA certificate for HTTPS proxy verification.
- `ols_config.proxy_config.no_proxy_hosts` -- List of hostnames that bypass the proxy.

## Constraints

1. At most one provider-specific configuration block (e.g., `openai_config`, `azure_openai_config`) may appear per provider entry. If more than one is present, the system must reject the configuration at startup.

2. The provider-specific configuration block must match the provider's `type`. If the type is `azure_openai` but the block present is `openai_config`, the system must reject the configuration.

3. Provider type must be one of the supported set. An unrecognized type string is a startup configuration error.

4. Every provider entry must have at least one model configured. Zero models is a startup configuration error.

5. Minimum TLS version when a security profile is configured must be `VersionTLS12` or higher. Lower versions must be rejected at configuration time.

6. Credentials are never logged. Parameters containing keys, tokens, or HTTP client objects must be redacted from log output.

7. The certificate store path is computed at startup and points to a PEM bundle file in the certificate directory. It is only used by OpenAI-family providers (OpenAI, Azure OpenAI, RHOAI vLLM, RHELAI vLLM).

8. Google Vertex providers require `credentials` to contain valid JSON representing a Google service account key. Non-JSON or non-object values must be rejected.

9. The default context window size is 128,000 tokens. The default max tokens for response is 4,096. These defaults apply when not explicitly configured, but may not be accurate for all models -- administrators should set model-specific values.

## Planned Changes

- [PLANNED: OLS-1680] Support AWS Bedrock as an LLM provider, enabling models hosted on Amazon's managed inference service.
- [PLANNED: OLS-2776] Support Anthropic as a direct LLM provider (not via Google Vertex), communicating with the Anthropic API natively.
- [PLANNED: OLS-1320] Support short-lived (rotating) tokens for all providers, replacing static API keys with tokens that are refreshed periodically.
- [PLANNED: OLS-1999] Support IBM WatsonX short-lived token authentication, enabling token-based auth that refreshes automatically rather than using a static API key.
