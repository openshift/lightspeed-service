# Agent Loop Test Harness

Test the OLS agent loop with realistic tool interactions — without an
OpenShift cluster.

## What This Provides

| Module | Purpose |
|---|---|
| `fake_tools.py` | Six fake OCP tools (`get_cluster_info`, `list_pods`, `describe_pod`, `drain_node`, `scale_deployment`, `delete_namespace`) with policy annotations |
| `conversation_fixtures.py` | Canned conversations at five sizes for compaction and context-window testing |
| `provider_matrix.py` | Parametrized cross-provider test utilities (OpenAI, Anthropic, Gemini) |
| `conftest.py` | Fixtures for tools, provider config, and LLM proxy credentials |

## Quick Start

### Offline (mocked LLM — no credentials needed)

```bash
uv run pytest tests/harness/ -v -k "not live"
```

### Live (real LLM calls via proxy)

```bash
export OLS_TEST_LLM_BASE_URL="https://your-proxy.example.com/v1"
export OLS_TEST_LLM_API_KEY="your-key"

# Single provider
uv run pytest tests/harness/ -v --provider=openai

# All providers
uv run pytest tests/harness/ -v --provider=all
```

### Override models per provider

```bash
export OLS_TEST_MODEL_OPENAI="gpt-4o"
export OLS_TEST_MODEL_ANTHROPIC="claude-sonnet-4-20250514"
export OLS_TEST_MODEL_GEMINI="gemini-2.5-pro"
```

## What This Tests

- **Tool classification** — ALLOW / DENY / CONFIRM policy routing
- **Tool execution** — coroutines return expected output and structured content
- **Oversized output** — `describe_pod` generates ~50 KB to trigger compaction
- **Conversation shapes** — fixtures exercise split-point logic, tool-result
  pair protection, and sacred-first-message preservation
- **Cross-provider edge cases** — parallel tool calls, text + tool in same
  response, empty `tool_args`

## Adding Fixtures

Add new conversations to `conversation_fixtures.py` and register them in
`CONVERSATION_FIXTURES`. Each fixture is a function returning
`list[dict[str, Any]]` with standard message fields (`role`, `content`,
and optionally `tool_calls`, `tool_call_id`).

## Architecture Notes

All interfaces use **plain dicts** — no LangChain types. Convert at the
boundary when wiring into `DocsSummarizer` or the streaming endpoint.
This keeps the harness portable to `lightspeed-stack` and other consumers.
