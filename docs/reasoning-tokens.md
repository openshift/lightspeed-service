# Reasoning Token Support for OpenAI Reasoning Models

**Status**: WIP / Draft
**PR**: [#2778](https://github.com/openshift/lightspeed-service/pull/2778)

## Problem

Reasoning models (GPT-5, o-series) produce an internal chain-of-thought (CoT)
alongside their final answer. OLS drops this today. During multi-round
tool-calling within a single request, the model has no memory of the reasoning
it did in the previous round — it only sees its final answer, not *why* it gave
it. This breaks multi-turn continuity for reasoning models.

Without passing reasoning back between rounds, the model:
- Re-reasons from scratch each round, producing repetitive verbose output
- Loops through multiple LLM invocations without converging
- Eventually times out

OpenAI explicitly recommends passing reasoning items back:
> "we highly recommend you pass back any reasoning items returned with the last
> function call"
> — [OpenAI Reasoning Guide](https://developers.openai.com/api/docs/guides/reasoning#keeping-reasoning-items-in-context)

## Scope

There are two distinct concerns:

1. **Backend correctness** — pass reasoning between tool-calling rounds so the
   model can maintain its chain of thought within a single request
2. **UI streaming** — surface reasoning summaries to the user so they can see
   the model's thinking in progress (like other AI chat products)

Reasoning is **ephemeral** — it is NOT stored in the conversation cache between
separate question/answer pairs. It only lives within the `messages` list during
a single request's tool-calling loop.

## Changes

### 1. OpenAI Responses API (`ols/src/llms/providers/openai.py`)

The default Chat Completions API does not expose reasoning tokens for GPT-5
(they are processed server-side and invisible). Switching to the Responses API
makes reasoning summaries available as content blocks.

For GPT-5 and o-series models:
- `reasoning={"effort": ..., "summary": ...}` — enables reasoning.
  `effort` controls reasoning depth. `summary` controls whether OpenAI
  generates a human-readable digest of the model's internal reasoning (the raw
  reasoning tokens are encrypted/opaque). These summaries are what get streamed
  to the UI as `event: reasoning`. Without `summary`, reasoning blocks still
  exist for context preservation but have no readable text to display.
- `verbosity` — controls how verbose the model's text output is

The `reasoning`, and `verbosity` parameters are whitelisted
in `OpenAIParameters` (`ols/src/llms/providers/provider.py`).

These are configurable per model in `olsconfig.yaml` under `parameters`:

```yaml
models:
  - name: gpt-5
    parameters:
      reasoning_effort: low      # low | medium | high (default: low)
      reasoning_summary: concise  # auto | concise | detailed (default: concise)
      verbosity: low             # low | medium | high (default: low)
```

When omitted, defaults are used. For non-reasoning models, these fields are
ignored. Validation is done in `ModelParameters`
(`ols/app/models/config.py`).

### 2. Reasoning extraction (`ols/src/query_helpers/docs_summarizer.py`)

The streaming loop in `iterate_with_tools` uses LangChain's `content_blocks`
property for **provider-agnostic** content extraction:

- `chunk.content_blocks` normalizes both Responses API (list of dicts) and
  Chat Completions API (string) into a standard `[{"type": ..., ...}]` format
- For OpenAI Responses API, the translator converts `{"type": "reasoning",
  "summary": [{"text": "..."}]}` → `{"type": "reasoning", "reasoning": "..."}`
- For Chat Completions, `"hello"` → `[{"type": "text", "text": "hello"}]`
- Blocks with `type: "reasoning"` → yield as `StreamedChunk(type="reasoning")`
- Blocks with `type: "text"` → yield as `StreamedChunk(type="text")`
- Works for any provider that registers a LangChain block translator
  (OpenAI, Anthropic, etc.)

> **LangGraph note:** `tool_calling_graph.py` uses LangGraph, which supports
> `stream_mode="messages"` with `version="v2"` for typed `StreamPart` output.
> The underlying chunks carry the same `content_blocks` property used here.
> If the graph path is refactored to stream directly to the caller,
> `stream_mode="messages"` would be the natural fit — each yielded
> `(AIMessageChunk, metadata)` tuple already has normalized `content_blocks`.

### 3. Reasoning passthrough between rounds (`ols/src/query_helpers/docs_summarizer.py`)

Previously, the AI message between rounds was:
```python
AIMessage(content="", tool_calls=tool_calls)  # reasoning lost
```

Now, all `AIMessageChunk`s from each round are accumulated and merged:
```python
accumulated = all_chunks[0]
for c in all_chunks[1:]:
    accumulated += c
AIMessage(
    content=accumulated.content,      # reasoning + text blocks
    tool_calls=tool_calls,
    additional_kwargs=accumulated.additional_kwargs,
)
```

This preserves the full content (reasoning blocks, text blocks, tool calls) so
the model can continue its chain of thought in the next round.

### 4. Token budget accounting (`ols/src/query_helpers/docs_summarizer.py`)

The AI message content (including reasoning) is now counted toward the tool
token budget. Previously only the `tool_calls` JSON was counted:
```python
ai_content_text = (
    json.dumps(ai_tool_call_message.content)
    if isinstance(ai_tool_call_message.content, list)
    else str(ai_tool_call_message.content)
)
ai_message_tokens = TokenHandler._get_token_count(
    token_handler.text_to_tokens(ai_content_text + json.dumps(tool_calls))
)
```

### 5. Loop termination for Responses API (`ols/src/query_helpers/docs_summarizer.py`)

The Responses API requires different loop termination logic than the Chat
Completions API:

- **Explicit exit on no tool calls**: The Chat Completions API sets
  `finish_reason="stop"` when the model finishes without tool calls, which
  triggers `stop_generation = True` → `return`. The Responses API does not set
  this, so the loop needs an explicit `if not tool_call_chunks: break` to avoid
  falling through to the next iteration.

- **No `chunk_position` stop detection**: The Responses API sends
  `chunk_position="last"` on the final chunk, but unlike `finish_reason` it
  fires for ALL completions (text-only and tool calls alike). Using it for
  early stop detection would kill tool-call processing. The `async for` loop
  ends naturally when the stream completes, which is sufficient.

- **Final-round tool binding with `tool_choice="none"`**: With Chat
  Completions, the final round removed tool definitions entirely and the model
  produced a clean text answer. With the Responses API, the conversation
  history contains `function_call` and `function_call_output` items from
  earlier rounds. When the model sees this history but no tool definitions in
  the request, it cannot produce proper `function_call` output items — instead
  it dumps the tool call arguments as `type: "text"` output items, polluting
  the answer with raw JSON. The fix: always bind tools in the final round but
  set `tool_choice="none"` to tell the model "don't call any tools, produce
  your final answer." This keeps the API context consistent with the
  conversation history while preventing further tool calls.

```python
# before (broken for Responses API):
llm = (
    self.bare_llm
    if is_final_round or not tools_map
    else self.bare_llm.bind_tools(tools_map)
)

# after:
if not tools_map:
    llm = self.bare_llm
elif is_final_round:
    llm = self.bare_llm.bind_tools(tools_map, tool_choice="none")
else:
    llm = self.bare_llm.bind_tools(tools_map)
```

### 6. New streaming event (`ols/app/endpoints/streaming_ols.py`)

- New constant `LLM_REASONING_EVENT = "reasoning"`
- `response_processing_wrapper` yields reasoning events for
  `StreamedChunk(type="reasoning")`
- `stream_event` formats reasoning:
  - `text/plain` → outputs reasoning text directly
  - `application/json` → `event: reasoning\ndata: {"id": N, "reasoning": "..."}`

### 7. StreamedChunk model (`ols/app/models/models.py`)

Added `"reasoning"` to the `type` literal:
```python
type: Literal["text", "tool_call", "tool_result", "end", "reasoning"]
```

### 8. Token counter (`ols/app/metrics/token_counter.py`, `ols/app/metrics/metrics.py`, `ols/app/models/models.py`)

The Responses API streaming path in `langchain-openai` passes `generation_chunk.text`
to `on_llm_new_token`. For Responses API models, this is the raw content list
(e.g. `[{'type': 'reasoning', 'summary': [...]}]` or `[{'type': 'text', 'text': '...'}]`)
instead of a string — a bug in `langchain-openai` where `ChatGenerationChunk.text`
is not always normalized to a string.

`on_llm_new_token` now handles both formats:
- **`str`**: counted as `output_tokens` (existing behavior for Chat Completions)
- **`list`**: parsed by `_extract_text_from_blocks` which separates content blocks
  by type — `type: "text"` blocks are counted as `output_tokens`, `type: "reasoning"`
  blocks have their `summary[].text` extracted and counted as `reasoning_tokens`.
  Both use tiktoken for counting, consistent with the existing mechanism.

Note: OpenAI's `usage_metadata.output_token_details.reasoning` reports *hidden
internal computation* tokens (always 0 when summaries are enabled), NOT the summary
text we stream. The summary text is what users see and is counted here via tiktoken.

New Prometheus metric `ols_llm_reasoning_token_total` (Counter, labels: provider,
model) tracks reasoning summary tokens separately from output tokens. The
`TokenCounter` dataclass has a new `reasoning_tokens: int = 0` field, and
`TokenMetricUpdater.__exit__` increments the metric.

### 9. Non-streaming path (`ols/src/query_helpers/docs_summarizer.py`)

`create_response` silently skips reasoning chunks (`pass`) to avoid
`ValueError` in the non-streaming code path.

## What is NOT changed

- **Conversation cache** — Reasoning is not stored. The response string
  accumulated by `streaming_ols.py` only includes `text` chunks, so reasoning
  is automatically excluded from the cache.
- **Non-OpenAI providers** — They continue using the Chat Completions path.
  No impact.
- **Non-reasoning OpenAI models (gpt-4o, etc.)** — No `reasoning` param is
  set, so they stay on Chat Completions. The structured content block handling
  falls through to the existing string-based path.

## Discovered issues during implementation

| Issue | Root cause | Fix |
|---|---|---|
| `KeyError: 'reasoning_content'` | Chat Completions API doesn't expose reasoning for GPT-5 | Switch to Responses API |
| `TypeError` in token counter | `langchain-openai` Responses API path passes raw content list as token | Parse list content blocks by type in `on_llm_new_token` — upstream bug in langchain-openai |
| Output tokens always 0 for Responses API | Text chunks also arrive as lists, not strings — skipped by `isinstance(token, str)` | `_extract_text_from_blocks` handles both `text` and `reasoning` block types |
| Model loops endlessly between rounds | Missing `break` when no tool calls after a round | `if not tool_call_chunks: break` |
| Model loops endlessly within a round | `finish_reason="stop"` never fires for Responses API | Remove `chunk_position="last"` stop check |
| Noisy `[thinking]` per-token in text/plain | Original text/plain formatting for reasoning | Output reasoning text directly |
| Excessive verbosity | `effort: "medium"` + no verbosity control | `effort: "low"` + `verbosity: "low"` |
| Reasoning not counted in token budget | Only `tool_calls` JSON was counted | Count `content` + `tool_calls` |
| Final round dumps tool args as text | Responses API model sees tool history but no tool defs → produces JSON as text blocks | Bind tools with `tool_choice="none"` in final round |

## Open items / TODOs


- [x] **Config-driven reasoning parameters** — `reasoning_effort`,
  `reasoning_summary`, and `verbosity` are configurable per model in
  `olsconfig.yaml` with validated values and hardcoded fallback defaults.
- [x] **Context pressure from reasoning** — `limit_conversation_history` runs
  once in `_prepare_prompt()` before the tool-calling loop starts. Messages
  added during the loop (AI messages with reasoning, tool results) are not
  subject to any further truncation — this was fine before because AI messages
  had `content=""`, but now they carry reasoning blocks. Reasoning content
  should not be truncated (it would break chain of thought). Short-term
  mitigation: reserve additional context headroom when reasoning is enabled
  (e.g. `max_rounds * avg_reasoning_tokens`). Long-term: proper mid-loop
  context management that drops older round messages — this is the same
  mechanism needed for [OLS-2685](https://issues.redhat.com/browse/OLS-2685)
  (tool call history between requests) and should be solved together.
- [x] **~~`use_previous_response_id`~~** — rejected. OpenAI keeps full
  conversation state server-side, making reasoning tokens an internal detail.
  OLS needs to control state management (token budgets, truncation, caching),
  cannot stream reasoning to the UI, and the approach is vendor-locked.
- [x] **`summary` default** — change default from `"auto"` to `"concise"` for
  token efficiency and predictable context usage. Users can override to
  `"auto"` or `"detailed"` via `reasoning_summary` in model config.
- [x] **Non-OpenAI reasoning models** — DeepSeek, Anthropic Claude have their
  own reasoning token formats. Current implementation is OpenAI-specific.
- [x] **UI integration** — frontend needs to handle `event: reasoning` SSE
  events and render them appropriately (collapsible thinking section, etc.).
- [ ] **Unit tests** — reasoning extraction, streaming, accumulation, token
  counting.
- [ ] **Integration tests** — end-to-end with a reasoning model.

## PR scoping

This work should be split into separate, independently reviewable PRs:

1. **Backend reasoning passthrough** — Responses API enablement, reasoning
   extraction, chunk accumulation between rounds, loop termination fixes, token
   counter resilience, token budget accounting for reasoning content. This is
   the core correctness fix (budget accounting must be included — without it,
   reasoning silently eats context window space and risks overflow).
2. **Configuration** — `reasoning_effort`, `reasoning_summary`, `verbosity`
   fields in `ModelParameters`, validation, provider reading. Standalone PR,
   no functional dependency on #1 (just uses the fields once #1 lands).
3. **Streaming to UI** — `StreamedChunk(type="reasoning")`, SSE
   `event: reasoning`, `stream_event` formatting. Can land with or after #1.
4. **Reasoning token metrics** — `reasoning_tokens` field in `TokenCounter`,
   `_extract_text_from_blocks` in `GenericTokenCounter`, new Prometheus metric
   `ols_llm_reasoning_token_total`. Standalone PR, no dependency on #1–#3.
5. **E2E test** - configuration and asserting reasoning events are sent and reasoning tokens computed

Each PR includes its own unit and integration tests.
