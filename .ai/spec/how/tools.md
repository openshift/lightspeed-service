# Tools -- Architecture

The tools subsystem manages MCP (Model Context Protocol) server integration,
tool discovery and filtering, human-in-the-loop approval gating, parallel tool
execution with retry/backoff, and token-budget-aware result truncation.

## Module Map

| File | Key symbols | Responsibility |
|---|---|---|
| `ols/utils/mcp_utils.py` | `build_mcp_config`, `gather_mcp_tools`, `get_mcp_tools`, `resolve_header_value`, `_normalize_tool_schema` | MCP client lifecycle: builds per-server transport configs with placeholder resolution, connects to servers via `MultiServerMCPClient`, gathers tools with fault isolation (one failing server does not block others), deduplicates by name (first-seen wins), and normalizes schemas for OpenAI compatibility. Optionally routes through ToolsRAG for query-based filtering. |
| `ols/src/tools/tools.py` | `execute_tool_calls_stream`, `enforce_tool_token_budget`, `execute_tool_call`, `_execute_with_retries`, `_extract_text_from_tool_output` | Tool execution engine: runs tool calls in parallel via `aiostream.merge`, applies retry/backoff for transient errors, extracts text from both string and content-block outputs, enforces per-tool and aggregate token budgets with a 3-tier truncation strategy. Emits typed streaming events (`ApprovalRequiredEvent`, `ToolResultEvent`). |
| `ols/src/tools/approval.py` | `need_validation`, `get_approval_decision`, `set_approval_decision`, `register_pending_approval`, `InMemoryPendingApprovalStore` | Approval state machine: determines whether a tool call requires user approval (based on config strategy and tool annotations), registers pending approvals in an in-memory store backed by `asyncio.Event`, waits for decisions with configurable timeout, and cleans up state on completion. |
| `ols/src/tools/tools_rag/hybrid_tools_rag.py` | `ToolsRAG`, `populate_tools`, `retrieve_hybrid`, `remove_tools` | Hybrid RAG for tool filtering: indexes tool name+description into Qdrant (in-memory) with dense embeddings and BM25 sparse vectors, retrieves relevant tools via reciprocal rank fusion (RRF) with configurable alpha weighting, grouped by server. Supports default servers (always included) and per-request client servers. |
| `ols/app/endpoints/tool_approvals.py` | `submit_tool_approval_decision` | REST endpoint (`POST /tool-approvals/decision`) for receiving user approval/rejection decisions. Delegates to `set_approval_decision` and returns 404/409 for missing or already-resolved approvals. |
| `ols/src/query_helpers/docs_summarizer.py` | `_resolve_tool_call_definitions`, `_process_tool_calls_for_round` | Caller-side orchestration: resolves LLM-emitted tool calls to executable definitions (validating name, args type, deduplication), invokes the execution engine, enforces the aggregate token budget, and emits streaming events for tool_call, approval_required, and tool_result. |

## Data Flow

### 1. MCP Client Setup (startup / per-request)

```
olsconfig.yaml (mcp_servers section)
  -> MCPServerConfig list with name, url, headers, timeout
  -> build_mcp_config(servers_list, user_token, client_headers)
     -> For each server:
        resolve_server_headers: substitute placeholders in header values
          "kubernetes" -> "Bearer {user_token}" (k8s service account token)
          "client"     -> value from client_headers[server_name][header_name]
          other        -> literal value (resolved at config load from file/env)
        Build MCPServerTransport dict: transport="streamable_http", url, headers, timeout
        Optionally attach custom CA bundle via httpx_client_factory
  -> MCPServersDict: {server_name -> transport config}
```

### 2. Tool Gathering (per-request or cached via ToolsRAG)

```
get_mcp_tools(query, user_token, client_headers)
  -> If no ToolsRAG configured:
       _gather_and_populate_tools -> gather_mcp_tools -> MultiServerMCPClient
       -> Connect to each server independently (fault-isolated)
       -> Collect StructuredTool list, tag each with metadata["mcp_server"]
       -> _normalize_tool_schema: add empty "properties"/{} to no-arg tools
       -> Deduplicate by name (first-seen wins, log warning for duplicates)
  -> If ToolsRAG configured:
       _populate_tools_rag (one-time for k8s servers, per-request for client servers)
       -> Index tools into Qdrant via populate_tools
       -> retrieve_hybrid(query, client_servers)
          -> Dense: encode query -> cosine similarity in Qdrant
          -> Sparse: BM25 over tokenized tool text (name + description)
          -> RRF fusion with alpha weight (default 0.8 = mostly dense)
          -> Filter by threshold, group by server
       -> Gather only the filtered tools from their source servers
```

### 3. Tool Binding and LLM Interaction

```
DocsSummarizer.__init__
  -> get_mcp_tools(query) returns list[StructuredTool]
  -> llm.bind_tools(tools) attaches tool schemas to LLM

LLM streaming response
  -> Chunks with tool_call_chunks accumulated until finish_reason
  -> tool_calls_from_tool_calls_chunks: merge partial chunks into complete calls
  -> _resolve_tool_call_definitions: validate each call
       Skip: missing name, duplicate name, unavailable tool, invalid args type
       Accept: (tool_id, tool_args, StructuredTool) triple
```

### 4. Tool Execution (per round)

```
_process_tool_calls_for_round
  -> Stream tool_call events to client
  -> execute_tool_calls_stream(tool_call_definitions, tools_token_budget)
     -> Split budget equally: per_tool_budget = total // len(calls)
     -> aiostream.merge: run all tool generators concurrently
        For each tool call:
          1. _evaluate_and_emit_approval_event
               need_validation(streaming, approval_type, tool_annotation)
                 -> "never": skip
                 -> "always": require approval
                 -> "tool_annotations": require unless readOnlyHint=true
               If needed: register_pending_approval -> yield approval_required event
                          -> await get_approval_decision (asyncio.Event + timeout)
                          -> If denied/timeout: yield rejection event, raise _ApprovalNotGrantedError
          2. _execute_with_retries
               Up to MAX_TOOL_CALL_RETRIES (2) + 1 attempts
               Backoff: 0.2s * 2^attempt (transient), 1.0s * 2^attempt (429 rate limit)
               On success: _extract_text_from_tool_output
                 -> String or content-block list (langchain-mcp-adapters >= 0.2.0)
                 -> Cheap char guard: tools_token_budget * 4 chars max
                 -> Cut at last newline boundary, append truncation warning
               On failure: return error status with truncated reason
          3. Yield ToolResultEvent with ToolMessage
  -> enforce_tool_token_budget(all_tool_messages, remaining_budget)
       Tier 1: char estimate (len/4) -- if < 90% of budget, skip tokenization
       Tier 2: precise tokenization -- if under budget, return as-is
       Tier 3: truncation
         If longest message can absorb excess (>= 2x excess): shrink only that one
         Otherwise: scale all messages proportionally (ratio = budget/total)
         Cut at last newline, append truncation warning
  -> Yield tool_result streaming events, charge token budget
  -> Append ToolMessages to conversation, loop back to LLM
```

## Key Abstractions

### MCP Client Wrapper

`MultiServerMCPClient` (from `langchain_mcp_adapters`) is the transport layer.
OLS wraps it with fault isolation: `gather_mcp_tools` connects to each server
individually so one unreachable server does not block others. Tool schemas are
normalized post-collection to ensure OpenAI compatibility (adding empty
`properties` dict to no-arg tools).

### Approval State Machine

States: `pending` -> `approved` | `rejected` | `timeout` | `error`

The store is pluggable via `PendingApprovalStoreBase` ABC, with
`InMemoryPendingApprovalStore` as the only implementation. Each pending approval
is a `PendingApproval` dataclass holding an `asyncio.Event` for the waiter and a
`decision: bool | None` field. The waiter blocks on `event.wait()` with
`asyncio.wait_for` for timeout enforcement. The `finally` block always deletes
the entry from the store, preventing memory leaks.

Three approval strategies (`ApprovalType` enum):
- `never` -- tools execute without approval (default)
- `always` -- every tool call requires user approval
- `tool_annotations` -- approval required unless the tool declares `readOnlyHint: true` in its MCP annotations

Approval is only supported on streaming requests (non-streaming always skips).

### Hybrid RAG for Tool Filtering

`ToolsRAG` extends `HybridRAGBase`, using an in-memory Qdrant instance. Tool
text is `name + description` (experimentally determined as best hit rate at
99.1%). Retrieval uses reciprocal rank fusion of dense cosine similarity and
BM25 sparse scores, with alpha=0.8 (mostly dense) and configurable top_k and
threshold. Results are grouped by server name, enabling server-scoped tool
gathering.

Default servers (k8s-auth) are always included in results; client-auth servers
are added per-request. The Qdrant upsert semantics mean re-indexing the same
tool updates rather than duplicates.

### 3-Tier Truncation Strategy

Tool outputs can be arbitrarily large. The system uses three tiers to balance
cost and precision:

1. **Char estimate** (`len / 4`): if total is clearly under 90% of the token
   budget, skip tokenization entirely.
2. **Precise tokenization**: only invoked when the char estimate is ambiguous.
   Tokenize each message to get exact counts.
3. **Proportional truncation**: if one message dominates (can absorb the excess
   while retaining half its content), shrink only that one. Otherwise scale all
   messages proportionally. Always cut at the last newline to avoid mid-line
   splits.

A separate pre-extraction char guard (`budget * 4 chars`) prevents tokenizing
arbitrarily large raw tool outputs before they even reach the budget enforcer.

## Integration Points

| Boundary | How it connects |
|---|---|
| **DocsSummarizer** | Calls `get_mcp_tools` during init to discover tools, binds them to LLM. Calls `execute_tool_calls_stream` and `enforce_tool_token_budget` during the tool-call loop. Emits `tool_call`, `approval_required`, and `tool_result` streaming events. |
| **Tool Approvals Endpoint** | `POST /tool-approvals/decision` receives `{approval_id, approved}` from the client. Calls `set_approval_decision` which sets the `asyncio.Event`, unblocking the waiter in `get_approval_decision`. |
| **TokenBudgetTracker** | DocsSummarizer charges tool results via `tracker.charge(TokenCategory.TOOL_RESULT, count)`. The tools_token_budget passed to execution is derived from the tracker's remaining budget. |
| **Streaming Protocol** | Three event types flow to the client: `StreamChunkType.TOOL_CALL` (before execution), `StreamChunkType.APPROVAL_REQUIRED` (when gated), `StreamChunkType.TOOL_RESULT` (after execution). Each carries structured data including tool name, args, output, truncation status, and optional structured_content from MCP artifacts. |
| **Config** | `MCPServers` / `MCPServerConfig` models define servers. `ToolsApprovalConfig` sets approval strategy and timeout. `ToolsRAGConfig` controls filtering parameters (alpha, top_k, threshold). All live under `OLSConfig`. |

## Implementation Notes

### Header Placeholder Resolution

MCP server headers support three resolution modes in `resolve_header_value`:
- `"kubernetes"` -- replaced with `"Bearer {user_token}"` from the request's k8s service account token
- `"client"` -- replaced with the value from `client_headers[server_name][header_name]`, passed in the request body
- Any other value -- used as-is (resolved at config load time, e.g., from file or environment variable)

If resolution fails (missing token or missing client header), the entire server
is skipped (`resolve_server_headers` returns `None`), and `build_mcp_config`
omits it from the config dict.

### Retry and Backoff

`_execute_with_retries` implements exponential backoff with two tiers:
- **Transient errors** (timeout, connection, OSError): base delay 0.2s, `0.2 * 2^attempt`
- **Rate-limit errors** (429, "rate limit", "too many requests"): base delay 1.0s, `1.0 * 2^attempt`
- Max retries: 2 (3 total attempts)
- Non-transient errors fail immediately without retry
- Error messages in tool results are truncated to 220 characters

### Parallel Tool Execution

When the LLM requests multiple tool calls in a single response, they execute
concurrently via `aiostream.stream.merge`. Each tool gets an equal share of the
token budget (`total_budget // num_calls`). Events (approval_required,
tool_result) are yielded as they arrive from any tool, not in request order.

### Tool Deduplication

Two levels of deduplication exist:
- **At gathering** (`_gather_and_populate_tools` with `deduplicate=True`): first-seen tool wins, duplicates from later servers are logged and dropped
- **At call resolution** (`_resolve_tool_call_definitions`): if a tool name appears in the `duplicate_tool_names` set, the call is skipped with a non-retryable error message

### Tool Schema Normalization

`_normalize_tool_schema` patches dict-based schemas (not Pydantic models) that
have `"type": "object"` but lack `"properties"`. This prevents `KeyError` in
LangChain's `BaseTool.args` and OpenAI's function-calling validation. Empty
`"required": []` is also added.

### Adding a New MCP Server

Adding a new MCP server requires only configuration changes, no code
modifications:

1. Add a server entry to the `mcp_servers.servers` list in `olsconfig.yaml`
   with `name`, `url`, optional `headers` (with placeholder values if needed),
   and optional `timeout`
2. If the server requires the user's k8s token, set the header value to
   `"kubernetes"`
3. If the server requires client-provided credentials, set the header value to
   `"client"` and document the required header names for API consumers
4. Restart the service; tools from the new server will be discovered
   automatically via `gather_mcp_tools`

### Custom CA Bundle

When `certificate_directory` is configured in `OLSConfig`, `build_mcp_config`
creates an `httpx.AsyncClient` factory with a custom SSL context pointing to the
CA bundle file. This factory is attached to every MCP server transport config,
enabling connections to servers with internal/self-signed certificates.

### Per-Round Timeout

Each tool-call round (LLM streaming + tool execution) is wrapped in
`asyncio.timeout(TOOL_CALL_ROUND_TIMEOUT)` (300 seconds) in the DocsSummarizer
loop. This is separate from the per-tool retry timeouts and the approval
decision timeout.
