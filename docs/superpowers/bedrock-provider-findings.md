# OLS-1680: AWS Bedrock Provider — Research Findings

## Summary

AWS Bedrock can be integrated into OLS using existing LangChain classes
(`ChatAnthropic`, `ChatOpenAI`) pointed at the `bedrock-mantle` endpoint.
No new LangChain provider package (`langchain-aws`) or `boto3` is required.

## Authentication

Bedrock API key — a simple Bearer token generated in the AWS Bedrock
console. Same key works for all models. Set via environment variable
or credentials file.

- 30-day long-term keys for dev (generated in console)
- Short-term keys via `aws-bedrock-token-generator` for production
- Environment variable: `AWS_BEARER_TOKEN_BEDROCK`

## Tested Models and LangChain Classes

All tested on 2026-06-18 with `bedrock-mantle.us-east-1.api.aws`.

### Claude (Anthropic Messages API)

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="anthropic.claude-opus-4-7",
    base_url="https://bedrock-mantle.{region}.api.aws/anthropic",
    api_key=BEDROCK_KEY,
)
```

- Endpoint: `/anthropic/v1/messages`
- Streaming: works
- Does NOT support Chat Completions or Responses API on Mantle

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

| Model family | LangChain class | Base URL path | Extra params |
|---|---|---|---|
| Claude | `ChatAnthropic` | `/anthropic` | — |
| OpenAI GPT-5.x | `ChatOpenAI` | `/openai/v1` | `use_responses_api=True` |
| DeepSeek, others | `ChatOpenAI` | `/v1` | — |

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

### Dependencies to add

- `langchain-anthropic` — explicit dependency (currently only transitive)

### No new dependencies needed

- `langchain-openai` — already in project
- `anthropic` — already in project
- `langchain-aws` / `boto3` — NOT needed

### Provider design

Two approaches, recommend **option A**:

**A. Single `bedrock` provider with model-based routing**

One provider type in `olsconfig.yaml`. The provider detects the model
prefix and picks the right LangChain class:

- `anthropic.*` → `ChatAnthropic`
- `openai.*` → `ChatOpenAI` with `use_responses_api=True`
- Everything else → `ChatOpenAI` (Chat Completions)

**B. Separate providers per API**

- `bedrock_anthropic` → `ChatAnthropic`
- `bedrock_openai` → `ChatOpenAI`

Option A is simpler for users (one provider type, one set of credentials).

### Config shape

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

### Files to create/modify

| Step | File | Action |
|---|---|---|
| 1 | `ols/constants.py` | Add `PROVIDER_BEDROCK = "bedrock"` |
| 2 | `ols/src/llms/providers/bedrock.py` | New provider class |
| 3 | `ols/src/llms/providers/provider.py` | Add parameter sets |
| 4 | `ols/app/models/config.py` | Add `BedrockConfig` (region) |
| 5 | `pyproject.toml` | Add `langchain-anthropic` |
| 6 | `tests/unit/llms/providers/test_bedrock.py` | Unit tests |
| 7 | `tests/unit/llms/providers/test_providers.py` | Registration test |

### Jira stories

- **OLS-1895** — OLS service support for AWS Bedrock (this repo)
- **OLS-2605** — OLSConfig CR support for Bedrock (operator repo)
