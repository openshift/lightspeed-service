# Query Pipeline -- Architecture

The query pipeline is the core request-processing path that transforms a user question into an LLM-generated answer with optional RAG context, conversation history, skill injection, and multi-round tool calling. `DocsSummarizer` is the single orchestrator that owns this entire flow.

## Module Map

### `ols/app/endpoints/ols.py` -- Non-streaming entry point

- `process_request()` -- Auth, redaction, attachment appending, quota check. Returns a `ProcessedRequest` dataclass.
- `generate_response()` -- Constructs `DocsSummarizer`, calls either `create_response()` (sync) or `generate_response()` (async generator). Catches `PromptTooLongError` and generic LLM errors, translating them to HTTP status codes.
- `store_conversation_history()` -- Persists the completed turn (query + response + tool data) to the conversation cache.
- `store_transcript()` -- Writes a JSON transcript file to disk when collection is enabled.
- `consume_tokens()` -- Deducts input/output tokens from all configured quota limiters.

### `ols/app/endpoints/streaming_ols.py` -- Streaming entry point

- `conversation_request()` -- FastAPI endpoint that returns a `StreamingResponse` wrapping the async generator.
- `response_processing_wrapper()` -- Async generator that consumes `StreamedChunk` objects from `DocsSummarizer.generate_response()`, converts each to an SSE-formatted string, accumulates the full response text, then calls `store_data()` and `consume_tokens()` after the stream ends.
- `stream_event()` -- Formats a single chunk as either plain text or SSE JSON (`data: {...}\n\n`) depending on the requested `media_type`.
- `stream_start_event()` / `stream_end_event()` -- Bookend events for JSON media type streams.

### `ols/src/query_helpers/docs_summarizer.py` -- The orchestrator

- `DocsSummarizer(QueryHelper)` -- Central class. Constructed once per request.
  - `__init__()` -- Loads LLM, resolves MCP tool servers, creates `TokenBudgetTracker`.
  - `generate_response()` -- Async generator: the main pipeline stages (RAG, skill, history, prompt, tool loop). Yields `StreamedChunk` objects.
  - `create_response()` -- Sync wrapper that drains `generate_response()` into a `SummarizerResponse`.
  - `_prepare_prompt_context()` -- Builds a template prompt to measure base token cost, retrieves RAG nodes, truncates them to fit.
  - `_build_final_prompt()` -- Assembles the real prompt with history, RAG, and skill content. Checks total against budget.
  - `iterate_with_tools()` -- Multi-round tool-calling loop. Each round: invoke LLM, collect chunks, process tool calls, feed results back.
  - `_collect_round_llm_chunks()` -- Streams one LLM invocation, separates text/reasoning chunks from tool-call chunks.
  - `_process_tool_calls_for_round()` -- Resolves tool calls, executes them, emits `TOOL_CALL` and `TOOL_RESULT` streaming events.
  - `_invoke_llm()` -- Binds tools to the LLM (or unbinds on final round) and streams via LangChain's `chain.astream()`.

### `ols/src/query_helpers/query_helper.py` -- Base class

- `QueryHelper` -- Resolves provider/model defaults, loads system prompt by mode (`ASK` vs `TROUBLESHOOTING`), stores `llm_loader` callable.

### `ols/src/query_helpers/history_support.py` -- History compression

- `prepare_history()` -- Async generator: retrieves cache entries, decides whether compression is needed, yields progress events and final `(history, truncated)` tuple.
- `compress_conversation_history()` -- Splits entries into a "keep" tail and a "summarize" prefix, calls `summarize_entries()`, rewrites the cache with a synthetic summary entry.
- `summarize_entries()` -- Calls the LLM to produce a conversational summary. Retries transient failures up to 3 times with exponential backoff.
- `_split_entries_by_token_budget()` -- Walks entries newest-to-oldest, accumulating token counts until the budget is exhausted.

### `ols/src/query_helpers/attachment_appender.py` -- Attachment formatting

- `append_attachments_to_query()` -- Concatenates formatted attachment blocks to the query string.
- `format_attachment()` -- Wraps content in Markdown code fences with language hint. YAML attachments get an intro message identifying the resource kind/name.

### `ols/utils/token_handler.py` -- Token accounting

- `TokenHandler` -- Stateless tokenizer wrapper (tiktoken `cl100k_base`). Methods: `text_to_tokens()`, `tokens_to_text()`, `truncate_rag_context()`, `limit_conversation_history()`, `calculate_and_check_available_tokens()`.
- `TokenBudgetTracker` -- Per-request stateful budget tracker. Tracks usage by `TokenCategory` enum (PROMPT, HISTORY, RAG, SKILL, TOOL_DEFINITIONS, AI_ROUND, TOOL_RESULT).
- `PromptTooLongError` -- Raised when any stage exceeds the available budget.

## Data Flow

### 1. Request intake (endpoint layer)

```
HTTP POST /query or /streaming_query
  -> process_request()
       auth -> user_id, user_token
       redact_query() -> PII-scrubbed query
       retrieve_attachments() -> redact_attachments()
       append_attachments_to_query() -> query now includes attachment blocks
       validate_requested_provider_model()
       check_tokens_available() -> quota gate
  -> generate_response()
       constructs DocsSummarizer(provider, model, mode, user_token, client_headers, streaming)
       calls create_response() [sync] or generate_response() [async generator]
```

### 2. DocsSummarizer.__init__

```
QueryHelper.__init__() -> resolve provider/model defaults, load system prompt
_prepare_llm()         -> load LLM via llm_loader, store provider_config/model_config
build_mcp_config()     -> resolve MCP server configs with user token/headers
TokenBudgetTracker()   -> initialize with context_window_size, max_response_tokens,
                          max_tool_tokens, round_cap_fraction
set_tool_loop_max_rounds() -> store max iterations for adaptive budget
```

### 3. generate_response() stages

```
Stage 1: RAG context
  _prepare_prompt_context(query, rag_retriever)
    -> build template prompt, count base prompt tokens, charge PROMPT
    -> retrieve RAG nodes, truncate to history_budget, charge RAG

Stage 2: Skill selection
  skills_rag.retrieve_skill(query) -> (skill, confidence)
  skill.load_content()
  if skill_tokens > available * 0.8: skip skill
  else: charge SKILL, yield SKILL_SELECTED chunk

Stage 3: History retrieval and compression
  prepare_history() -> async generator
    -> retrieve cache entries
    -> if compression enabled and entries overflow budget:
         yield HISTORY_COMPRESSION_START
         compress_conversation_history()
         yield HISTORY_COMPRESSION_END
    -> yield (history_messages, truncated)
  charge HISTORY for each message

Stage 4: Final prompt assembly
  _build_final_prompt(query, history, rag_chunks, skill_content)
    -> GeneratePrompt(...).generate_prompt(model)
    -> budget overflow check (including tool_definitions_tokens)

Stage 5: Tool resolution
  get_mcp_tools(query, user_token, client_headers)
  count tool definition tokens, check against prompt budget

Stage 6: Tool-calling loop (iterate_with_tools)
  for round 1..max_rounds:
    _collect_round_llm_chunks() -> text/reasoning chunks + tool_call chunks
    yield text/reasoning StreamedChunks
    if no tool calls or final round: break
    _process_tool_calls_for_round():
      resolve tool call definitions (skip duplicates, missing, invalid)
      yield TOOL_CALL chunks
      execute tools within round budget
      yield TOOL_RESULT chunks
      append AI message + tool messages to prompt for next round
      charge AI_ROUND and TOOL_RESULT

Stage 7: Finalization
  yield END chunk with rag_chunks, truncated flag, token_counter
```

### 4. Post-generation (endpoint layer)

```
Non-streaming:
  store_conversation_history() -> cache
  store_transcript() -> filesystem
  consume_tokens() -> quota limiters
  return LLMResponse

Streaming:
  response_processing_wrapper() consumes the async generator:
    yield stream_start_event (JSON mode)
    for each StreamedChunk: yield formatted SSE event
    on END chunk: extract rag_chunks, truncated, token_counter
    store_data() -> cache + transcript
    consume_tokens() -> quota limiters
    yield stream_end_event with referenced docs and available quotas
```

## Key Abstractions

### DocsSummarizer as orchestrator

`DocsSummarizer` is the only class that knows the full pipeline sequence. It inherits from `QueryHelper` for provider/model/prompt defaults but adds all pipeline logic. Both the sync (`create_response`) and streaming paths funnel through the same `generate_response()` async generator -- the sync path simply drains it.

### Token budget partitioning

The context window is divided into three non-overlapping slices:

```
context_window_size = prompt_budget + max_response_tokens + max_tool_tokens

prompt_budget = context_window_size - max_response_tokens - max_tool_tokens
```

Within `prompt_budget`, categories are charged in order: PROMPT (base template) -> RAG -> SKILL -> HISTORY. The `history_budget` property returns the remaining space after prompt, RAG, and skill tokens are deducted:

```
history_budget = prompt_budget - usage[PROMPT] - usage[RAG] - usage[SKILL]
```

Token counts use tiktoken's `cl100k_base` encoding with a 10% safety buffer (`TOKEN_BUFFER_WEIGHT = 1.1`, applied via `ceil(len(tokens) * 1.1)`).

The tool execution budget is separate from the prompt budget. Within it, each round is capped at `round_cap_fraction` (default 0.6, configurable 0.3-0.8) of the remaining tool budget:

```
tools_round_budget = int(tool_budget_remaining * round_cap_fraction)
```

The adaptive `tools_round_execution_budget()` additionally considers horizon (rounds remaining) to distribute budget more evenly:

```
exec_budget = min(
    int(remaining_tool * round_cap_fraction),
    remaining_tool // max(1, tool_rounds_left)
)
```

Tool definitions tokens (serialized JSON schemas of all MCP tools) are charged under `TOOL_DEFINITIONS` but excluded from the tool execution budget calculation -- only `AI_ROUND` and `TOOL_RESULT` consume the execution pool.

### StreamedChunk types and interleaving during tool calling

All pipeline output flows through `StreamedChunk(type, text, data)`. The `StreamChunkType` enum defines:

| Type | When emitted | Payload |
|---|---|---|
| `TEXT` | LLM text tokens | `text` field |
| `REASONING` | Model reasoning tokens (e.g. OpenAI o-series) | `text` field |
| `TOOL_CALL` | Before tool execution | `data`: tool name, args, id, server metadata |
| `APPROVAL_REQUIRED` | Tool requires user approval | `data`: approval request details |
| `TOOL_RESULT` | After tool execution | `data`: id, name, status, content, round, structured_content |
| `SKILL_SELECTED` | After skill retrieval | `data`: name, confidence, optional skipped/reason |
| `HISTORY_COMPRESSION_START` | Before LLM-based compression | `data`: `{"status": "started"}` |
| `HISTORY_COMPRESSION_END` | After compression completes | `data`: status, duration_ms |
| `END` | Pipeline complete | `data`: rag_chunks, truncated, token_counter |

During a multi-round tool call, the interleaving pattern is:

```
[round 1]
  TEXT/REASONING chunks (partial answer or reasoning)
  TOOL_CALL chunk (intent to call tool X)
  TOOL_CALL chunk (intent to call tool Y)
  TOOL_RESULT chunk (result from tool X)
  TOOL_RESULT chunk (result from tool Y)
[round 2]
  TEXT/REASONING chunks (LLM incorporates tool results)
  ... more TOOL_CALL/TOOL_RESULT if needed ...
[final round]
  TEXT chunks (final answer, tools unbound or tool_choice="none")
END chunk
```

### History compression

When `history_compression_enabled` is true, `prepare_history()` uses a two-phase approach:

1. **Budget check**: `_split_entries_by_token_budget()` walks entries newest-to-oldest with an effective budget of `available_tokens * 0.85` (`HISTORY_TOKEN_BUDGET_RATIO`). If all entries fit, no compression occurs.

2. **Compression**: When entries overflow, `compress_conversation_history()` keeps the most recent `entries_to_keep` entries (default 5) verbatim. All older entries are passed to `summarize_entries()`, which calls the LLM with a structured summarization prompt.

3. **Retry logic**: `summarize_entries()` retries up to 3 attempts with exponential backoff (1s, 2s, 4s). Each attempt has a 20-second timeout (`SUMMARY_ATTEMPT_TIMEOUT_SECONDS`). Only transient errors (timeout, connection, rate limit, 429/502/503) trigger retries; other errors fail immediately.

4. **Fallback**: If summarization fails entirely, the system falls back to keeping only the newest entries (or the single most recent entry if none fit). The compressed history is persisted back to the cache as a synthetic `[Previous conversation summary]` entry followed by the kept raw entries.

When compression is disabled, `limit_conversation_history()` does simple newest-first truncation to fit the token budget.

## Integration Points

### LLM providers

`DocsSummarizer` loads the LLM via `llm_loader` (default: `load_llm` from `ols.src.llms.llm_loader`). The bare LLM is used directly for streaming (`chain.astream()`) and for history summarization. Tools are bound per-round via LangChain's `bind_tools()` with `strict=False` to avoid Responses API issues.

### Conversation cache

Read path: `_retrieve_previous_input()` calls `config.conversation_cache.get()` to load full conversation history as `CacheEntry` objects.

Write path: `store_conversation_history()` calls `config.conversation_cache.insert_or_append()`. During compression, `_rewrite_cache()` deletes then re-inserts the compressed entries.

### RAG index

`_prepare_prompt_context()` calls `rag_retriever.retrieve(query)` (LlamaIndex `BaseRetriever`). Results are filtered by `RAG_SIMILARITY_CUTOFF` (0.3) and truncated by `truncate_rag_context()` to fit the remaining token budget. Each accepted node becomes a `RagChunk(text, doc_url, doc_title)`.

### MCP tools

`build_mcp_config()` resolves server configurations from `config.mcp_servers.servers` with user token/headers. `get_mcp_tools()` fetches available tools from all configured MCP servers. Tool execution goes through `execute_tool_calls_stream()`, which yields `APPROVAL_REQUIRED` and `TOOL_RESULT` events.

### Skills

`config.skills_rag.retrieve_skill(query)` returns a matched skill and confidence score. The skill's content is loaded from disk and injected into the prompt via `GeneratePrompt`. If the skill would consume more than 80% of the available history budget, it is skipped.

### Quota

Pre-check: `check_tokens_available()` calls `quota_limiter.ensure_available_quota()` before processing. Post-response: `consume_tokens()` deducts the actual input/output token counts from all configured quota limiters and records usage in `token_usage_history`.

### Prompts

`GeneratePrompt` (from `ols.src.prompts.prompt_generator`) constructs the `ChatPromptTemplate` from the system prompt, RAG context, history, tool-calling flag, query mode, cluster version, and optional skill content. The system prompt is selected by mode: `QUERY_SYSTEM_INSTRUCTION` for ASK, `TROUBLESHOOTING_SYSTEM_INSTRUCTION` for TROUBLESHOOTING.

## Implementation Notes

### Token budget calculation algorithm

The budget is calculated once in `DocsSummarizer.__init__` via `TokenBudgetTracker` and charged incrementally as each stage runs. The order matters: prompt template tokens are measured first (using a dummy prompt with placeholder history/RAG), then RAG tokens, then skill tokens, and finally history tokens. Each stage uses `history_budget` (the remaining headroom) as its ceiling, which naturally shrinks as prior stages consume tokens. If any stage would push total usage beyond `prompt_budget`, a `PromptTooLongError` is raised.

### How streaming chunks interleave during tool calling

In non-final rounds, the LLM is invoked with tools bound (no `tool_choice` constraint). `_collect_round_llm_chunks()` accumulates all chunks from one LLM invocation, separating tool-call chunks from text/reasoning chunks. Text chunks are yielded immediately for streaming. After the LLM stream ends, `_process_tool_calls_for_round()` emits `TOOL_CALL` events, executes the tools, then emits `TOOL_RESULT` events. The AI message (including any reasoning content) and all tool messages are appended to the prompt template for the next round.

On the final round (round == max_rounds or no tools available), the LLM is invoked with `tool_choice="none"` and `strict=False` to force a text-only response. The Granite model family requires special handling: the first 6 chunks of a tool call (`"", "<", "tool", "_", "call", ">"`) are suppressed via `skip_special_chunk()` to avoid leaking tool-call markup to the user.

### History compression retry logic

`summarize_entries()` uses 3 attempts with delays of 1s, 2s (exponential backoff, `delay *= 2`). Each LLM call is wrapped in `asyncio.wait_for()` with a 20-second timeout. Only transient errors (containing keywords like "timeout", "connection", "rate limit", "429", "502", "503") trigger retries. Non-transient errors (e.g., auth failures, malformed responses) break immediately. If all attempts fail, `compress_conversation_history()` falls back to keeping only the newest raw entries without a summary.

### Relationship between generate_response() and the streaming generator

`generate_response()` is an async generator that yields `StreamedChunk` objects. For streaming endpoints, this generator is passed directly to `response_processing_wrapper()`, which consumes it chunk-by-chunk, formats each as SSE, and yields strings to FastAPI's `StreamingResponse`. For non-streaming endpoints, `create_response()` calls `run_async_safely()` around an inner `drain_generate_response()` coroutine that collects all chunks into a single `SummarizerResponse`. This means the same pipeline code runs for both paths -- the only difference is whether chunks are streamed to the client or buffered internally.

### How attachments are positioned in the prompt

Attachments are appended to the query string before it enters `DocsSummarizer`. The `append_attachments_to_query()` function concatenates each attachment after the original query text, separated by blank lines. Each attachment is wrapped in Markdown code fences with a language hint derived from its content type. YAML attachments additionally get an intro sentence like "For reference, here is the full resource YAML for Pod 'my-pod':". The modified query (with attachments) is what the LLM sees as the user message; the original query (without attachments) is preserved separately for transcript storage.

### Round cap fraction for tool calling

The `round_cap_fraction` (default 0.6, range 0.3-0.8) limits how much of the remaining tool budget any single round can consume. This prevents an early round from exhausting all tool tokens, leaving nothing for subsequent rounds. The fraction is applied to `tool_budget_remaining`, which tracks only execution traffic (`AI_ROUND` + `TOOL_RESULT` categories) -- tool definition tokens are charged separately and do not reduce the execution pool. If the remaining budget for a round falls below `MIN_TOOL_EXECUTION_TOKENS` (100), all tool calls for that round are skipped with an error message. Each LLM round also has a hard timeout of 300 seconds (`TOOL_CALL_ROUND_TIMEOUT`).

### Max iterations by mode

The tool-calling loop cap depends on the query mode: ASK defaults to 5 rounds, TROUBLESHOOTING defaults to 15 rounds. An explicit `max_iterations` config value can raise but never lower the mode-specific default. The final round always forces `tool_choice="none"` to guarantee a text answer.
