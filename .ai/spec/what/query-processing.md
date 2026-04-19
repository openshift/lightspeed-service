# Query Processing

The query processing pipeline transforms a user request into an LLM-generated
response by orchestrating validation, context assembly, generation, and
post-processing across eight sequential stages.

## Behavioral Rules

### Stage 1: Request Validation and Redaction

1. The system must authenticate the user and extract a user identity before
   any further processing.

2. If the request includes a conversation ID, the system must validate its
   format and reject the request if invalid. If no conversation ID is
   provided, the system must generate one.

3. The system must apply all configured query filters (regex-based PII
   redaction) to the user query text before any logging or downstream
   processing occurs.

4. If the request specifies a provider or model, the system must validate
   them against the configured provider/model list and reject the request
   if either is unknown.

5. The system must verify that the user has available quota before
   proceeding. If quota is exhausted, the request must be rejected.

### Stage 2: Attachment Processing

6. Each attachment must declare a supported attachment type and content
   type. The system must reject the request if any attachment uses an
   unsupported type or content type.

7. All attachment content must be redacted through the same PII filters as
   the query text before any further processing.

8. Each attachment must be formatted as a Markdown code block with a
   language tag matching its content type (yaml, json, xml, or no tag for
   plain text).

9. For YAML attachments, the system must attempt to parse the content and
   extract the resource `kind` and `metadata.name`. If both are found,
   the system must prepend a contextual introduction identifying the
   resource. If parsing fails, a generic introduction must be used.

10. Formatted attachments must be appended directly to the query text
    before any other pipeline stage (RAG retrieval, history, prompt
    composition).

11. The original query text (without attachments) must be preserved
    separately for transcript storage.

### Stage 3: Token Budget Calculation and RAG Retrieval

12. The system must compute the prompt budget by subtracting the response
    token reservation and (when tool calling is enabled) the tool token
    reservation from the model's context window size.

13. The system must construct a temporary prompt to measure the base prompt
    token cost (system prompt, message structure). If the base prompt alone
    exceeds the prompt budget, the system must raise a PromptTooLongError.

14. If a RAG index is available, the system must retrieve relevant document
    chunks using the query. Retrieved chunks must be filtered by a
    similarity score cutoff -- chunks below the threshold are rejected.
    Chunks must be truncated to fit within the remaining token budget
    after the base prompt cost. For details on RAG retrieval, see
    `what/rag.md`.

15. If no RAG index is configured, the system must proceed without
    reference context and log a warning.

16. Token counts for every budget category (prompt, RAG, history, skill,
    tool definitions, AI rounds, tool results) must be tracked through a
    unified per-request budget tracker.

### Stage 4: History Retrieval and Compression

17. If no user ID or conversation ID is provided, the system must skip
    history entirely and use an empty history.

18. If a conversation ID is provided, the system must retrieve the full
    conversation history from the cache. For details on cache behavior,
    see `what/conversation-history.md`.

19. When history compression is disabled in configuration, the system must
    apply simple token-based truncation, keeping the newest messages that
    fit within the available budget and dropping the oldest.

20. When history compression is enabled, the system must compare the
    conversation history against an effective history budget (a fixed
    fraction of the available token budget). If all entries fit, no
    compression occurs.

21. When history overflows the effective budget, the system must compress
    by summarizing older entries into a single synthetic entry using the
    LLM, preserving a configurable number of recent entries verbatim.
    Summarization must be retried on transient errors with exponential
    backoff. If summarization fails entirely, the system must fall back
    to the verbatim entries that fit.

22. During history compression, the system must emit a
    `history_compression_start` streaming event before compression begins
    and a `history_compression_end` event after it completes.

### Stage 5: Skill Selection

23. If a skills index is configured, the system must match the query
    against available skills using hybrid RAG retrieval and return the
    best match with a confidence score. For details on skill matching,
    see `what/skills.md`.

24. If a skill is selected, the system must load its content. If loading
    fails (filesystem error), the system must fall back to no skill
    without failing the request.

25. If the selected skill's token cost exceeds 80% of the remaining token
    budget (after prompt and RAG), the system must skip the skill and emit
    a `skill_selected` event with `skipped: true` and the reason. If the
    skill uses more than 50% but fits within 80%, the system must log a
    warning but still use the skill.

26. When a skill is successfully selected, the system must emit a
    `skill_selected` streaming event with the skill name and confidence
    score.

### Stage 6: Prompt Composition

27. The system must assemble the final prompt from the following components,
    in this order within the system message:
    - Base system prompt (mode-dependent; see `what/agent-modes.md`)
    - Agent instructions (appended when tool calling is enabled)
    - Contextual instructions for RAG context (appended when RAG chunks
      are present)
    - History instructions (appended when history is present)
    - Skill instructions and content (appended when a skill is selected)
    - RAG context (appended at the end of the system message)

28. After the system message, conversation history (if present) must be
    included as a message placeholder, followed by the user query as the
    final human message.

29. After assembling the prompt, the system must validate that the total
    token usage (prompt + RAG + history + skill + tool definitions) does
    not exceed the prompt budget. If it does, the system must raise a
    PromptTooLongError.

### Stage 7: LLM Generation with Tool Calling Loop

30. The system must resolve available tools from configured MCP servers
    for the current request. Tool definitions must be serialized and their
    token count charged against the tool budget.

31. The system must send the composed prompt to the LLM and stream the
    response. Each round of LLM invocation may yield text chunks,
    reasoning chunks, and/or tool call chunks.

32. If the LLM requests tool calls, the system must resolve each call to
    an executable tool, execute the calls, and append the results to the
    conversation for the next round. The system must then re-invoke the
    LLM. This loop continues until the LLM produces a final text response
    or the iteration limit is reached. For details on iteration limits,
    see `what/agent-modes.md`. For details on tool execution, approval,
    and filtering, see `what/tools.md`.

33. On the final iteration, the system must invoke the LLM without tool
    bindings (or with tool_choice=none) to force a text-only answer,
    guaranteeing termination.

34. Each LLM invocation round must be subject to a per-round timeout. If
    the timeout is reached, the system must return a user-facing error
    message and terminate the loop.

35. During the tool calling loop, the system must emit `tool_call`
    streaming events when the LLM requests tool calls and `tool_result`
    events when tool execution completes.

36. If a tool execution error occurs that cannot be recovered, the system
    must emit an error message to the user and terminate the loop.

### Stage 8: Response Storage and Quota Consumption

37. After the response is fully generated, the system must store the
    conversation turn (query + response + attachments + tool interactions)
    in the conversation cache. For details on cache behavior, see
    `what/conversation-history.md`.

38. If transcript collection is enabled, the system must store a transcript
    record containing metadata (provider, model, user, conversation ID,
    mode, timestamp), the redacted query, the LLM response, RAG chunks
    used, tool interactions, and attachments.

39. The system must deduct input and output tokens from the user's quota
    if quota limiting is configured. Token usage must also be recorded in
    the usage history if configured. For details on quota, see
    `what/quota.md`.

40. The response must include referenced documents derived from RAG chunks,
    token counts (input and output), and the user's remaining quota.

## Token Budget System

41. The context window is partitioned into three top-level reserves:
    - **Response reserve**: tokens reserved for the LLM's generated
      response (`model.parameters.max_tokens_for_response`).
    - **Tool reserve**: tokens reserved for tool definitions and tool
      execution traffic (`model.max_tokens_for_tools`), computed as a
      configurable ratio of the context window
      (`model.parameters.tool_budget_ratio`). Set to zero when no MCP
      servers are configured.
    - **Prompt budget**: the remainder after subtracting the response and
      tool reserves. This budget is shared among the base prompt, RAG
      context, conversation history, and skill content.

42. Within the prompt budget, allocations are charged in order: base prompt
    first, then RAG, then skill content, then conversation history. Each
    stage consumes from the remaining prompt budget. History receives
    whatever budget remains after the prior stages.

43. Within the tool reserve, allocations are charged per round: tool
    definitions (charged once before the first round), then per-round AI
    message tokens and tool result tokens. A per-round cap limits how
    much of the remaining tool budget any single round can consume
    (`ols_config.tool_round_cap_fraction`).

44. Token counts are approximate, computed using a fixed tokenizer with a
    configurable buffer weight that inflates raw counts to reduce the risk
    of underestimation.

45. RAG chunks below a minimum token threshold are rejected to avoid
    injecting fragments too small to be useful.

## Streaming vs Non-Streaming

46. Both streaming and non-streaming endpoints execute the same eight-stage
    pipeline using the same orchestrator. The difference is in delivery:
    - **Streaming**: Returns a StreamingResponse that emits chunks
      incrementally as SSE events (JSON format) or plain text. The client
      receives text tokens, reasoning blocks, tool call intents, tool
      results, skill selection events, and history compression events as
      they occur.
    - **Non-streaming**: Runs the pipeline to completion internally,
      collecting all chunks. Text chunks are concatenated into a single
      response string. The complete response is returned as a single JSON
      object.

47. Response storage, transcript recording, and quota consumption happen
    after the response is fully generated in both modes.

48. Error handling differs by mode: streaming returns errors as stream
    events within the response body; non-streaming raises HTTP exceptions
    with appropriate status codes.

## Error Handling

49. If the user query (with base prompt overhead) exceeds the prompt
    budget, the system must raise a PromptTooLongError. In non-streaming
    mode this results in HTTP 413. In streaming mode this is emitted as
    an error event in the stream.

50. If tool definitions together with the current prompt exceed the prompt
    budget, the system must raise a PromptTooLongError with a message
    identifying the tool definitions as the cause.

51. If history compression fails, the system must degrade gracefully by
    using the entries that fit in budget without compression, rather than
    failing the request.

52. If skill loading fails, the system must fall back to no skill without
    failing the request.

53. If a tool execution round fails with an unrecoverable error, the
    system must emit an error message and terminate the tool loop, but
    still return whatever text has been generated so far.

54. If an LLM invocation round times out, the system must return a
    user-facing timeout message and terminate the loop.

## Configuration Surface

| Field Path | Type | Default | Purpose |
|---|---|---|---|
| `ols_config.query_filters[]` | list | None | Regex-based PII redaction filters applied to queries and attachments |
| `ols_config.history_compression_enabled` | bool | true | Enable/disable LLM-based history compression |
| `ols_config.max_iterations` | int | None | Override tool-calling iteration cap (see `what/agent-modes.md`) |
| `ols_config.tool_round_cap_fraction` | float | (see constants) | Fraction of remaining tool budget usable per round |
| `ols_config.system_prompt_path` | string | None | Override the default system prompt (see `what/agent-modes.md`) |
| `model.parameters.max_tokens_for_response` | int | (per model) | Tokens reserved for LLM response generation |
| `model.parameters.tool_budget_ratio` | float | (see constants) | Fraction of context window reserved for tool traffic |
| `model.context_window_size` | int | (per model) | Total context window size for the model |
| `ols_config.user_data_collection.transcripts_disabled` | bool | false | Disable transcript storage |
| `ols_config.conversation_cache` | object | None | Cache backend configuration (see `what/conversation-history.md`) |

## Constraints

1. **Pipeline order is fixed.** The eight stages must execute in the
   defined order. Later stages depend on budget calculations and context
   assembled by earlier stages.

2. **Redaction before logging.** No query or attachment content may be
   logged or processed before PII redaction is applied.

3. **Token budget is a hard limit.** The total prompt token usage must
   never exceed the prompt budget. Any stage that would cause an overflow
   must either truncate its contribution or raise PromptTooLongError.

4. **Tool budget is separate from prompt budget.** Tool definitions and
   tool execution traffic draw from the tool reserve, not the prompt
   budget. However, if tool definitions cause the combined total to
   exceed the prompt budget, the request fails.

5. **Response and tool reserves are subtracted first.** The prompt budget
   is always computed as context window minus response reserve minus tool
   reserve. These reserves are non-negotiable.

6. **History gets the residual budget.** History is allocated whatever
   remains after prompt, RAG, and skill have been charged. It is never
   given a fixed allocation that could starve other components.

7. **Attachments are part of the query.** Once attachments are formatted
   and appended, they are indistinguishable from the query text for all
   downstream stages (token counting, prompt composition, storage).

8. **Conversation storage is best-effort.** If storing the conversation
   turn in the cache fails, the system logs the error and raises an HTTP
   500. The response has already been delivered in streaming mode.

## Planned Changes

| Jira Key | Summary |
|---|---|
| OLS-2825 | Consolidate context window token budget management into a single, well-documented module |
| OLS-2840 | Refactor DocsSummarizer: extract ToolCallingAgent class to separate prompt orchestration from tool loop execution |
| OLS-2898 | Raise `max_iterations` to 50 and validate orchestration loop early-exit behavior (see also `what/agent-modes.md`) |
