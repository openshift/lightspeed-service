# LLM Provider Implementation Guide

Read this when adding a new LLM provider or modifying an existing one.

## Structure

All providers live in `ols/src/llms/providers/`. Each provider is a single file named after the provider type (e.g. `openai.py`, `watsonx.py`).

## Required Steps for a New Provider

### 1. Register the provider type constant

Add the provider type string to `ols/constants.py`:

```python
PROVIDER_MYPROVIDER = "my_provider"
SUPPORTED_PROVIDER_TYPES = frozenset({
    ...
    PROVIDER_MYPROVIDER,
})
```

### 2. Define allowed parameters

In `ols/src/llms/providers/provider.py`, add a `ProviderParameter` set and a generic-to-LLM parameter mapping:

```python
MyProviderParameters = {
    ProviderParameter("api_key", str),
    ProviderParameter("model", str),
    ProviderParameter("temperature", float),
    ProviderParameter("max_tokens", int),
}

MyProviderParametersMapping: dict[str, str] = {
    GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: "max_tokens",
}
```

Then register both in the module-level dicts:

```python
available_provider_parameters[PROVIDER_MYPROVIDER] = MyProviderParameters
generic_to_llm_parameters[PROVIDER_MYPROVIDER] = MyProviderParametersMapping
```

### 3. Implement the provider class

```python
@register_llm_provider_as(constants.PROVIDER_MYPROVIDER)
class MyProvider(LLMProvider):

    url: str = "https://default.api.endpoint"
    credentials: Optional[str] = None

    @property
    def default_params(self) -> dict[str, Any]:
        self.url = str(self.provider_config.url or self.url)
        self.credentials = self.provider_config.credentials
        return {
            "api_key": self.credentials,
            "model": self.model,
            "temperature": 0.01,
            "max_tokens": 512,
        }

    def load(self) -> BaseChatModel:
        return SomeLangchainChatModel(**self.params)
```

Key rules:
- The `@register_llm_provider_as` decorator handles registry — no manual registration needed
- `default_params` must read credentials and URL from `self.provider_config`
- `load()` must return a `BaseChatModel` or `LLM` (LangChain type)
- Use `self._construct_httpx_client(True, False)` / `(True, True)` for sync/async HTTP clients
- Provider-specific config (e.g. `provider_config.openai_config`) takes precedence over generic config

### 4. Add provider-specific config model (if needed)

If the provider has unique config fields, add a config class in `ols/app/models/config.py`:

```python
class MyProviderConfig(ProviderSpecificConfig, extra="forbid"):
    credentials_path: str
    some_specific_field: Optional[str] = None
```

Then wire it into `ProviderConfig` (search for `openai_config` to see the pattern).

### 5. Add provider-specific YAML config key to `ProviderConfig`

In `ols/app/models/config.py`, add the optional field and validation in `ProviderConfig`. Follow the `openai_config` pattern.

## Parameter Precedence

Parameters are applied in this order (later overrides earlier):

1. `default_params` property
2. Caller-supplied `params` argument
3. `config.dev_config.llm_params` (developer override, highest priority)

Unknown parameters (not in the provider's `ProviderParameter` set) are silently filtered out before `load()` is called.

## Testing a New Provider

- Add `tests/unit/llms/providers/test_myprovider.py`
- Use a `provider_config` fixture that constructs `ProviderConfig` from a dict — see `test_openai.py` for the pattern
- Required test cases:
  - `test_basic_interface` — verifies `load()` returns the correct LangChain type and `default_params` contains required keys
  - `test_params_handling` — verifies unknown params are filtered, known params pass through
  - `test_loading_provider_specific_parameters` — if provider-specific config exists
- Add the provider to `test_providers_are_registered` in `test_providers.py`
- Credentials for tests live in `tests/config/secret/apitoken`

## Existing Providers Reference

| Constant | File | LangChain class |
|---|---|---|
| `PROVIDER_OPENAI` | `openai.py` | `ChatOpenAI` |
| `PROVIDER_AZURE_OPENAI` | `azure_openai.py` | `AzureChatOpenAI` |
| `PROVIDER_RHOAI_VLLM` | `rhoai_vllm.py` | `ChatOpenAI` (OpenAI-compatible) |
| `PROVIDER_RHELAI_VLLM` | `rhelai_vllm.py` | `ChatOpenAI` (OpenAI-compatible) |
| `PROVIDER_WATSONX` | `watsonx.py` | `WatsonxLLM` |
| `PROVIDER_GOOGLE_VERTEX_ANTHROPIC` | `google_vertex.py` | `ChatAnthropicVertex` (Claude on Vertex) |
| `PROVIDER_GOOGLE_VERTEX` | `google_vertex.py` | `ChatGoogleGenerativeAI` |
| `PROVIDER_FAKE` | `fake_provider.py` | `FakeListLLM` / `FakeStreamingListLLM` |
| `PROVIDER_BEDROCK` | `bedrock.py` | `ChatBedrockConverse` / `ChatOpenAI` (model-prefix routing) |

Vertex AI in `olsconfig.yaml`: use `type: google_vertex_anthropic` with a `google_vertex_anthropic_config` block (`project`, `location`), or `type: google_vertex` with `google_vertex_config` (same shape). The file referenced by `credentials_path` may be either a **service account key** (`type: service_account`) or **user OAuth JSON** (`type: authorized_user` with `refresh_token`, `client_id`, and `client_secret`). Each file must include the `type` field. Credentials are scoped for Vertex with the cloud-platform OAuth scope.

AWS Bedrock in `olsconfig.yaml`: use `type: bedrock` with `url` set to the Bedrock Mantle gateway (e.g. `https://bedrock-mantle.us-east-1.api.aws`). Model names must use the fully qualified Bedrock model ID (e.g. `anthropic.claude-opus-4-7`, `openai.gpt-5.4`, `deepseek.v3.1`) — the provider prefix determines which LangChain class to use:

- `anthropic.*` → `ChatBedrockConverse` via native Bedrock Converse API (region prefix auto-prepended)
- `openai.*` → `ChatOpenAI` via Responses API
- All others → `ChatOpenAI` via Chat Completions API

Two authentication methods are supported:

1. **Bearer token** — store a Bedrock API key in `credentials_path` (standard `apitoken` file). The key is set via `AWS_BEARER_TOKEN_BEDROCK` env var for `ChatBedrockConverse` and passed as `openai_api_key` for `ChatOpenAI`.

2. **IAM credentials** — set `credentials_path` to a directory containing `aws_access_key_id` and `aws_secret_access_key` files. Optional `role_arn` file triggers STS `assume_role()`. For `ChatBedrockConverse`, a pre-configured boto3 client is passed directly; for `ChatOpenAI`, SigV4 signing is injected via `httpx-aws-auth`.

Fully qualified names are required because they match the Bedrock model IDs as reported by the AWS console and API. The region is extracted from the Mantle URL. See `docs/superpowers/bedrock-provider-findings.md` for research details.
