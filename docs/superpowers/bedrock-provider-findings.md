# OLS-1680: AWS Bedrock Provider — Research Findings

## Summary

AWS Bedrock is integrated into OLS using `ChatBedrockConverse` (for
Anthropic models) and `ChatOpenAI` (for OpenAI/DeepSeek) pointed at
the `bedrock-mantle` endpoint. Two authentication methods are supported:
Bearer token (Bedrock API key) and IAM credentials (SigV4).

## Authentication

Two authentication pathways are supported:

### Bearer token (Bedrock API key)

A simple Bearer token generated in the AWS Bedrock console. Same key
works for all models. Stored in a file referenced by `credentials_path`.

- 30-day long-term keys for dev (generated in console)
- Short-term keys via `aws-bedrock-token-generator` for production
- Environment variable: `AWS_BEARER_TOKEN_BEDROCK`

### IAM credentials (SigV4)

Standard AWS IAM credentials resolved via `boto3`. The provider reads
`aws_access_key_id` and `aws_secret_access_key` from a directory
specified in `credentials_path`. Optional `role_arn`
file triggers STS `assume_role()`.

- `ChatBedrockConverse` (Anthropic) — pre-configured boto3 `bedrock-runtime`
  client passed via `client` parameter (no env var mutation)
- `ChatOpenAI` (OpenAI/DeepSeek) — SigV4 signing via `httpx-aws-auth`
  injected into `http_client` / `http_async_client`

## Tested Models and LangChain Classes

All tested on 2026-06-18 with `bedrock-mantle.us-east-1.api.aws`.

### Claude (ChatBedrockConverse — implemented)

```python
from langchain_aws import ChatBedrockConverse
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = BEDROCK_KEY

llm = ChatBedrockConverse(
    model_id="us.anthropic.claude-opus-4-7",
    region_name="us-east-1",
)
```

- Uses native Bedrock Converse API (not Mantle HTTP endpoints)
- Streaming: works
- Region prefix (`us.`) auto-prepended from Mantle URL

Earlier testing also confirmed `ChatAnthropic` works via Mantle
(`/anthropic/v1/messages`), but `ChatBedrockConverse` was chosen for
better proxy/TLS handling and natural IAM credential support.

### OpenAI GPT-5.x (Responses API)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai.gpt-5.4",
    base_url="https://bedrock-mantle.{region}.api.aws/openai/v1",
    api_key=BEDROCK_KEY,
    use_responses_api=True,
)
```

- Endpoint: `/openai/v1/responses`
- Streaming: works
- Does NOT support Chat Completions (`/v1/chat/completions`)
- `use_responses_api=True` is required
- `max_output_tokens` minimum is 16

### DeepSeek (Chat Completions API)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek.v3.1",
    base_url="https://bedrock-mantle.{region}.api.aws/v1",
    api_key=BEDROCK_KEY,
)
```

- Endpoint: `/v1/chat/completions`
- Standard OpenAI-compatible, no special flags needed

## Endpoint Matrix

| Model family | LangChain class | API | Extra params |
|---|---|---|---|
| Claude | `ChatBedrockConverse` | Bedrock Converse (native) | region auto-extracted from URL |
| OpenAI GPT-5.x | `ChatOpenAI` | Mantle `/openai/v1` | `use_responses_api=True` |
| DeepSeek, others | `ChatOpenAI` | Mantle `/v1` | — |

## Available Models (in test account, us-east-1)

- `anthropic.claude-opus-4-7`
- `openai.gpt-5.4`
- `openai.gpt-5.5` (listed, not tested)
- `deepseek.v3.1`

Model list is account-specific. Use:
```bash
curl -s "https://bedrock-mantle.us-east-1.api.aws/v1/models" \
  -H "Authorization: Bearer $BEDROCK_KEY"
```

## What Did NOT Work

- `ChatOpenAI` without `use_responses_api=True` for GPT-5.x → 400 error
- `ChatOpenAI` for Claude (neither Chat Completions nor Responses) → 400
- `ChatBedrockConverse` for GPT-5.x → invalid model ID (Converse API
  doesn't support OpenAI models)
- `ChatBedrockConverse` for Claude without `us.` prefix → needs
  inference profile ID (`us.anthropic.claude-opus-4-7`)
- Converse API (`bedrock-runtime`) → requires different model IDs,
  different auth patterns; unnecessary when using `bedrock-mantle`

## Implementation Plan for OLS

### Dependencies added

- `langchain-aws` — for `ChatBedrockConverse` (Anthropic route)
- `httpx-aws-auth` — for SigV4 signing (IAM auth for ChatOpenAI)

### No new dependencies needed

- `langchain-openai` — already in project
- `boto3` — transitive dependency of `langchain-aws`

### Provider design

Single `bedrock` provider with model-prefix routing. The provider
detects the model prefix and picks the right LangChain class:

- `anthropic.*` → `ChatBedrockConverse` via native Bedrock Converse API
- `openai.*` → `ChatOpenAI` with `/openai/v1` and `use_responses_api=True`
- Everything else → `ChatOpenAI` with `/v1` (Chat Completions)

**Why not separate providers (like `google_vertex` / `google_vertex_anthropic`)?**

Bedrock is a single gateway with one set of credentials. Splitting into
`bedrock_anthropic`, `bedrock_openai`, etc. would force users to
duplicate URL and credential config for each model family. The single
provider keeps config simple — one entry covers all Bedrock models.

**Why fully qualified model names (e.g. `anthropic.claude-opus-4-7`)?**

Model names in config use the Bedrock model ID exactly as AWS reports
them (e.g. in the Bedrock console, `oras discover` output, and the
`/v1/models` API response). The provider prefix (`anthropic.`, `openai.`,
`deepseek.`) is part of the Bedrock model identity — it determines which
API endpoint the model supports. Using fully qualified IDs avoids
maintaining a separate model-to-family mapping and ensures the value
passed to the API is always correct.

### Config shape

Bearer token auth:

```yaml
llm_providers:
  - name: my-bedrock
    type: bedrock
    url: https://bedrock-mantle.us-east-1.api.aws
    credentials_path: /path/to/bedrock_api_key
    models:
      - name: anthropic.claude-opus-4-7
      - name: openai.gpt-5.4
      - name: deepseek.v3.1
```

IAM auth (credentials_path points to a directory):

```yaml
llm_providers:
  - name: my-bedrock
    type: bedrock
    url: https://bedrock-mantle.us-east-1.api.aws
    credentials_path: /path/to/aws_creds_dir/
    # directory must contain:
    #   aws_access_key_id
    #   aws_secret_access_key
    #   role_arn  (optional — for STS assume_role)
    models:
      - name: anthropic.claude-opus-4-7
      - name: openai.gpt-5.4
```

### Files to create/modify

| Step | File | Action |
|---|---|---|
| 1 | `ols/constants.py` | Add `PROVIDER_BEDROCK = "bedrock"` |
| 2 | `ols/src/llms/providers/bedrock.py` | New provider class |
| 3 | `ols/src/llms/providers/provider.py` | Add parameter sets |
| 4 | `ols/app/models/config.py` | Add Bedrock IAM fields to `ProviderConfig` |
| 5 | `pyproject.toml` | Add `langchain-aws`, `httpx-aws-auth` |
| 6 | `tests/unit/llms/providers/test_bedrock.py` | Unit tests |
| 7 | `tests/unit/llms/providers/test_providers.py` | Registration test |

## IAM / STS Credential Support (implemented)

The Mantle gateway supports **AWS SDK credential chain** auth (SigV4)
in addition to Bearer tokens.

**Reference**: [Get started with OpenAI GPT-5.5, GPT-5.4 models, and Codex
on Amazon Bedrock](https://aws.amazon.com/blogs/aws/get-started-with-openai-gpt-5-5-gpt-5-4-models-and-codex-on-amazon-bedrock/)

### How it works

1. `boto3.Session()` resolves credentials from `aws_access_key_id`
   and `aws_secret_access_key` files in the `credentials_path` directory
2. If `role_arn` file exists, `sts.assume_role()` is called first
3. For Anthropic: a pre-configured boto3 `bedrock-runtime` client is
   passed to `ChatBedrockConverse` via the `client` parameter
4. For OpenAI/DeepSeek: SigV4 signing is injected via `httpx-aws-auth`
   into `ChatOpenAI`'s `http_client` / `http_async_client`

### Dependencies

- `httpx-aws-auth` — SigV4 signing for httpx
- `boto3` — transitive dependency of `langchain-aws`

### Bedrock IAM config fields on `ProviderConfig`

Read from `credentials_path` directory (same pattern as Azure):

- `aws_access_key_id` — AWS access key
- `aws_secret_access_key` — AWS secret key
- `role_arn` — STS role to assume (required when Bedrock permissions
  are granted to a role rather than directly to the IAM user)
- Region is derived from the Mantle URL

### Tested: all three routes work with IAM credentials

Tested on 2026-06-22 with IAM user `Bedrock-ci-user`.

| Model family | Class | IAM auth mechanism |
|---|---|---|
| `anthropic.*` | `ChatBedrockConverse` | boto3 client (SigV4, no env vars) |
| `openai.*` | `ChatOpenAI` | `httpx-aws-auth` SigV4 handler |
| `deepseek.*` | `ChatOpenAI` | `httpx-aws-auth` SigV4 handler |

### ChatBedrockConverse for Anthropic models (implemented)

`ChatBedrockConverse` (`langchain-aws`) works with Bearer token auth
via the `AWS_BEARER_TOKEN_BEDROCK` env var — no AWS SDK credentials
(access key / secret key) required.  Tested successfully with
`us.anthropic.claude-opus-4-7`.

Reference: [langchain-aws#582](https://github.com/langchain-ai/langchain-aws/issues/582)
— supported since `langchain-aws>=0.2.28`.

Advantages over `ChatAnthropic` + Mantle:
- Proxy and TLS handled natively by boto3 (`proxies`, `ca_bundle`)
- Temperature parameter accepted (gracefully ignored with warning
  if model doesn't support it, instead of 400 error)
- Natural path to STS support (boto3 credential chain)

The provider passes the configured API key directly to
`ChatBedrockConverse` via the `bedrock_api_key` constructor parameter.
The model ID is auto-prefixed with the region prefix extracted from the
Mantle URL (e.g. `anthropic.claude-opus-4-7` → `us.anthropic.claude-opus-4-7`).

OpenAI/DeepSeek models continue using `ChatOpenAI` via Mantle, as
`ChatBedrockConverse` does not support them.

### Jira stories

- **OLS-1895** — OLS service support for AWS Bedrock (this repo)
- **OLS-2605** — OLSConfig CR support for Bedrock (operator repo)
