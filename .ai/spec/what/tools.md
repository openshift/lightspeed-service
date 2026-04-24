# Tools

MCP (Model Context Protocol) tool integration enables the LLM to call external
tools -- such as querying cluster state, performing actions, or fetching data --
making OLS context-aware against the user's actual cluster rather than relying
solely on general knowledge.

## Behavioral Rules

### MCP Server Configuration

1. Each MCP server is defined with a unique name, URL, optional per-server
   timeout, and optional authorization headers. Server names must be unique
   within the configuration; duplicate names must be rejected at config load
   time.

2. Header values support three resolution modes:
   - `"kubernetes"`: replaced at runtime with the user's Kubernetes bearer
     token (formatted as `Bearer {token}`).
   - `"client"`: replaced with a header value provided by the client in the
     API request body, keyed by server name and header name.
   - Any other value: treated as a literal (typically resolved from a file
     path at config load time).

3. If header resolution fails for a server (e.g., the user has no token for a
   `"kubernetes"` header, or the client omitted a required `"client"` header),
   that server must be silently skipped -- the request must not fail.

4. The system must expose an API that tells clients which MCP servers require
   client-provided headers and which header names are required, so clients can
   supply them in requests. [PLANNED: OLS-2684 -- remove client MCP headers]

5. When a certificate directory is configured, all MCP connections must use
   the service's custom CA bundle for TLS verification.

### Tool Gathering

6. Tools must be collected from all configured MCP servers at query time.
   Gathering is fault-isolated per server: if one MCP server is unreachable
   or errors, tools from other servers must still be returned.

7. Each gathered tool must carry metadata indicating which MCP server it came
   from.

8. Tool schemas must be normalized for LLM compatibility: object-type schemas
   lacking a `properties` key must have an empty `properties` dict and empty
   `required` list added.

9. When the same tool name appears from multiple servers, the first
   occurrence wins (deduplication). Duplicate tools from later servers must
   be logged and dropped.

### Tool Execution

10. When the LLM requests multiple tool calls in a single round, all calls
    must execute concurrently. Results are streamed as they complete from any
    tool, not in request order. For how tool execution integrates with the
    generation loop, see `what/query-processing.md` (stage 7).

11. Transient errors (timeouts, connection resets, temporary failures) must be
    retried up to 2 times (3 total attempts) with exponential backoff: base
    delay 0.2 seconds, doubled on each retry.

12. Rate-limit errors (HTTP 429, "too many requests") must use a longer base
    delay of 1.0 second with the same exponential backoff and retry count.

13. Non-transient errors must fail immediately without retry. On final
    failure (retries exhausted or non-transient), the error message must be
    passed to the LLM as a tool result with error status. Error messages in
    tool results must be truncated to 220 characters.

14. Each tool-call round (LLM generation + tool execution) is subject to a
    per-round timeout (`TOOL_CALL_ROUND_TIMEOUT` = 300 seconds). For
    iteration limits across rounds, see `what/agent-modes.md`.

### Token Budget for Tool Results

15. A configurable fraction of the model's context window is reserved for tool
    traffic (definitions + execution results), controlled by
    `model.parameters.tool_budget_ratio` (default 0.25, range 0.10--0.60).
    This reserve is computed at startup as
    `context_window_size * tool_budget_ratio` and set to zero when no MCP
    servers are configured. For how the tool reserve integrates with the
    overall token budget, see `what/query-processing.md` (token budget
    system).

16. When multiple tool calls execute in parallel within a round, the
    remaining tool token budget must be split equally among the calls.

17. Tool outputs must be truncated using a 3-tier strategy:
    - **Tier 1 -- character estimate**: estimate token count at ~4 characters
      per token. If the estimate is clearly under 90% of the remaining
      budget, skip tokenization entirely.
    - **Tier 2 -- precise tokenization**: if the character estimate is
      ambiguous, tokenize each tool output to get exact counts. If the total
      is within budget, return outputs unmodified.
    - **Tier 3 -- proportional truncation**: if the longest output alone can
      absorb the excess (its half is >= the excess), shrink only that output.
      Otherwise, scale all outputs proportionally to fit the budget. All
      truncation must cut at the last newline boundary to avoid mid-line
      splits, and append a truncation warning.

18. Before tokenization, a cheap character-level size guard
    (`budget * 4 chars`) must be applied to raw tool output to prevent
    the CPU cost of tokenizing arbitrarily large responses. Strings are cut
    at the last newline boundary before the character limit.

### Tool Filtering via Hybrid RAG

19. When `ols_config.tool_filtering` is configured, the system must use
    hybrid RAG (dense embeddings + sparse BM25) to select relevant tools
    based on the user's query. For the underlying hybrid RAG mechanism, see
    `what/rag.md`.

20. Tools must be indexed by their name and description (concatenated as
    search text). Retrieval uses reciprocal rank fusion (RRF) of dense
    cosine similarity and BM25 sparse scores.

21. The dense-vs-sparse weight is controlled by `tool_filtering.alpha`
    (default 0.8; 1.0 = full dense, 0.0 = full sparse). Results below
    `tool_filtering.threshold` (default 0.01) must be excluded.

22. Results must be grouped by MCP server. When any tool from a server passes
    the threshold, all selected tools from that server are gathered together.

23. Default servers (those using kubernetes-token authentication) and any
    client-auth servers with headers provided in the current request must
    always be included in the search scope, regardless of RAG score.

24. Kubernetes-auth server tools are populated into the RAG index once and
    cached; client-auth server tools are populated per request when client
    headers are provided.

25. If tool filtering fails (RAG error), the system must fall back to
    returning all tools from all servers rather than returning an empty set.

26. If tool filtering is not configured, all tools from all servers must be
    passed to the LLM without filtering.

### Tool Approval Workflow

27. Three approval strategies are supported, selected by
    `tools_approval.strategy`:
    - `never` (default): all tool calls execute immediately without approval.
    - `always`: every tool call requires explicit user approval.
    - `tool_annotations`: approval is required by default, but tools that
      declare `readOnlyHint: true` in their MCP annotation metadata are
      exempt.

28. Approval is only supported for streaming requests. Non-streaming requests
    must never trigger the approval flow, regardless of the configured
    strategy.

29. When approval is required for a tool call:
    a. The system must emit an `approval_required` streaming event to the
       client containing a unique approval ID, the tool name, description,
       arguments, and annotation metadata.
    b. The system must then block and wait for the client to submit an
       approval or rejection via the `POST /tool-approvals/decision`
       endpoint.
    c. If approved, the tool executes normally through the retry policy.
    d. If rejected, a non-retryable error result is returned to the LLM
       with a message instructing it not to retry the call.
    e. If the configured timeout expires (default 600 seconds, controlled by
       `tools_approval.approval_timeout`), the call is treated as timed out
       with a non-retryable error.

30. Each approval request receives a unique ID. Approval state is stored in
    memory per process and is not persisted across restarts. After a decision
    is applied (or times out), the approval state must be cleaned up
    immediately to prevent memory leaks.

31. The approval decision endpoint must return HTTP 404 if no pending
    approval exists for the given ID, and HTTP 409 if the approval was
    already resolved. A decision for an already-resolved approval must not
    change the outcome.

### MCP Apps

32. MCP servers can serve `ui://` URI resources containing HTML/JS/CSS for
    rendering in the console UI as app iframes.

33. The console can proxy tool calls to a specific MCP server outside the
    normal LLM conversation flow for interactive app functionality.

34. Both resource fetching and direct tool calls require the same
    authentication and header resolution as normal MCP tool calls.

35. Structured content returned by tool calls (e.g., rich data for UI
    rendering) must be preserved in tool result metadata and forwarded to
    the client.

## Configuration Surface

| Field Path | Type | Default | Purpose |
|---|---|---|---|
| `mcp_servers.servers[]` | list | [] | MCP server definitions |
| `mcp_servers.servers[].name` | string | required | Unique server identifier |
| `mcp_servers.servers[].url` | string | required | Server HTTP endpoint |
| `mcp_servers.servers[].timeout` | int | 5 | Per-server request timeout in seconds |
| `mcp_servers.servers[].headers` | map | {} | Authorization headers (values are file paths, `"kubernetes"`, or `"client"`) |
| `model.parameters.tool_budget_ratio` | float | 0.25 | Fraction of context window reserved for tool traffic (0.10--0.60) |
| `ols_config.tool_round_cap_fraction` | float | 0.6 | Fraction of remaining tool budget usable per round (0.3--0.8) |
| `ols_config.tool_filtering` | object | none | Enables hybrid RAG tool filtering when present |
| `ols_config.tool_filtering.embed_model_path` | string | none | Path to sentence transformer model for embeddings |
| `ols_config.tool_filtering.alpha` | float | 0.8 | Dense vs sparse retrieval weight (0.0--1.0) |
| `ols_config.tool_filtering.top_k` | int | 10 | Number of tools to retrieve (1--50) |
| `ols_config.tool_filtering.threshold` | float | 0.01 | Minimum similarity score (0.0--1.0) |
| `tools_approval.strategy` | enum | `never` | Approval strategy: `never`, `always`, or `tool_annotations` |
| `tools_approval.approval_timeout` | int | 600 | Seconds to wait for user approval decision (>= 1) |

## Constraints

1. **Server names are unique.** Duplicate MCP server names in configuration
   must be rejected at startup. Tool names are deduplicated at gathering time
   (first-seen wins).

2. **Fault isolation is mandatory.** A single unreachable MCP server must
   never prevent tools from other servers from being gathered or executed.

3. **Tool budget is separate from prompt budget.** Tool definitions and tool
   execution traffic draw from the tool reserve, not the prompt budget.
   See `what/query-processing.md` for the full budget partitioning.

4. **Approval requires streaming.** The approval workflow is architecturally
   dependent on the streaming protocol for client communication. Non-streaming
   requests bypass approval entirely.

5. **Approval state is ephemeral.** Pending approval state is in-memory per
   process. A process restart clears all pending approvals; clients will
   receive no response for approvals that were pending at restart time.

6. **Truncation preserves line boundaries.** All truncation (both the
   per-tool character guard and the aggregate budget enforcer) must cut at
   newline boundaries to avoid delivering partial lines to the LLM.

7. **Retry policy is fixed.** The retry count (2 retries), base delays
   (0.2s transient, 1.0s rate-limit), and exponential backoff formula are
   compile-time constants, not configurable. Only the tool budget ratio,
   round cap fraction, and approval timeout are user-configurable.

8. **RAG filtering is optional.** When `ols_config.tool_filtering` is absent,
   all tools are passed to the LLM. When present but failing, the system
   falls back to all tools. The system must never return an empty tool set
   due to a filtering infrastructure failure.

## Planned Changes

| Jira Key | Summary |
|---|---|
| OLS-2685 | Previous tools in context -- carry tool results from prior conversation turns into the current prompt |
| OLS-2715 | ocp-mcp additional toolsets -- expand the default MCP server set |
| OLS-2684 | Remove client MCP headers -- eliminate the `"client"` header placeholder mechanism |
| OLS-2491 | MCP client improvements -- transport and reliability enhancements |
| OLS-1797 | Block sensitive tool args -- reject tool calls whose arguments match blocked patterns before execution |
