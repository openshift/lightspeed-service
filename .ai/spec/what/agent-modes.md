# Agent Modes

The service supports two agent modes -- ASK and TROUBLESHOOTING -- that control
the system prompt persona, agent instructions, and tool-calling iteration depth
for each request. This file is the single source of truth for mode-specific
behavior; other specs reference it rather than redefining mode rules.

## Behavioral Rules

### Mode Selection

1. The mode is selected per request via the `mode` field on the API request
   body (`LLMRequest.mode`). Mode is per-request, not per-session; different
   requests in the same conversation may use different modes.

2. The `mode` field accepts two values defined by the `QueryMode` enum: `ask`
   (default) and `troubleshooting`. If omitted, the mode defaults to `ask`.

### ASK Mode

3. ASK mode serves general question-and-answer interactions about OpenShift
   and related Red Hat products. The persona is a friendly, authoritative
   technical expert focused on documentation knowledge.

4. In ASK mode, the base system prompt is `QUERY_SYSTEM_INSTRUCTION`. This
   prompt defines the "OpenShift Lightspeed" expert persona, scopes expertise
   to OpenShift and listed Red Hat products, and instructs the LLM to
   prioritize provided context and chat history as the primary source of truth.

5. When tool calling is enabled (MCP servers are configured), ASK mode
   appends agent instructions to the system prompt. For most models,
   `AGENT_INSTRUCTION_GENERIC` is used; for Granite-family models,
   `AGENT_INSTRUCTION_GRANITE` is used instead. Both variants are followed
   by the shared `AGENT_SYSTEM_INSTRUCTION` block (tool-call deduplication
   rules and concise style guide).

6. ASK mode uses a default maximum tool-calling iteration limit of 5
   (`DEFAULT_MAX_ITERATIONS`).

### TROUBLESHOOTING Mode

7. TROUBLESHOOTING mode serves diagnostic and remediation interactions
   focused on live cluster issues. The persona is a specialist in OpenShift
   troubleshooting and diagnostics.

8. In TROUBLESHOOTING mode, the base system prompt is
   `TROUBLESHOOTING_SYSTEM_INSTRUCTION`. This prompt includes the live
   cluster's OpenShift version (injected via the `{cluster_version}`
   placeholder) and defines three response structures based on query type:
   diagnosis (evidence, root cause, fix), assessment (state summary, anomaly
   flagging, severity ranking), and question (direct answer).

9. The cluster version is retrieved from the Kubernetes API at request time
   when the mode is TROUBLESHOOTING. In ASK mode the cluster version is not
   retrieved and is set to a sentinel value.

10. When tool calling is enabled, TROUBLESHOOTING mode appends
    `TROUBLESHOOTING_AGENT_INSTRUCTION` followed by
    `TROUBLESHOOTING_AGENT_SYSTEM_INSTRUCTION` to the system prompt. These
    instructions define:
    - A structured investigation protocol: scope affected resources, gather
      evidence (logs, events, metrics, pods), cross-reference related
      resources, follow causality chains, find root cause, correlate with
      recent changes, and provide fixes or mitigations.
    - Tool usage rules: verify arguments, no duplicate calls with same
      arguments, build a complete picture before concluding, always check
      logs even for "Running" pods, never guess pod names, sample up to 3
      representative pods per deployment, never ask the user to run commands.
    - A metrics investigation workflow: `get_alerts` first, then
      `list_metrics` (with `name_regex`), then `get_label_names`, then
      `get_label_values`, then `execute_instant_query` or
      `execute_range_query`. Steps must not be skipped, and metric names must
      not be guessed.

11. TROUBLESHOOTING mode uses a default maximum tool-calling iteration limit
    of 15 (`DEFAULT_MAX_ITERATIONS_TROUBLESHOOTING`).

### Iteration Limit Resolution

12. The effective iteration limit for a request is resolved as follows:
    - Each mode has a built-in default: ASK = 5, TROUBLESHOOTING = 15.
      These defaults are stored in `MAX_ITERATIONS_BY_MODE`.
    - The administrator may set `ols_config.max_iterations` in the
      configuration file to raise the cap. The effective limit is the greater
      of the configured value and the mode's built-in default. The config
      value can raise the cap but never lower it below the mode default.
    - When `ols_config.max_iterations` is not set (None), the mode default
      is used as-is.

13. On the final iteration of the tool-calling loop (when the round index
    equals the effective max), the LLM is invoked without tool bindings (or
    with `tool_choice="none"`) to force a text-only answer. This guarantees
    termination.

### Common Behavior

14. Both modes share the same token budget system (`TokenBudgetTracker`).
    The context window is partitioned across system prompt, RAG context,
    conversation history, tool definitions, tool outputs, and response
    tokens. Mode does not change the budget partitioning rules.

15. Both modes use the same RAG retrieval pipeline. Retrieved document
    chunks are injected into the prompt identically regardless of mode.

16. Both modes use the same conversation history handling: history is loaded
    from the conversation cache, truncated to fit the available token budget,
    and optionally compressed (when `ols_config.history_compression_enabled`
    is true).

17. Both modes use the same skill selection pipeline. A matched skill's
    prompt content is injected into the system prompt regardless of mode.

18. Both modes use the same streaming response format (`StreamedChunk` with
    types: text, tool_call, tool_result, skill_selected,
    history_compression_start, history_compression_end, reasoning, end).

19. Both modes share the same tool token budget ratio
    (`tool_budget_ratio`), tool round cap fraction
    (`tool_round_cap_fraction`), and per-round tool execution timeout
    (`TOOL_CALL_ROUND_TIMEOUT` = 300 seconds).

### System Prompt Selection

20. The system prompt for a request is determined by the following priority:
    1. If `dev_config.enable_system_prompt_override` is true and the request
       includes a `system_prompt` field, use the request-supplied prompt.
    2. Else if `ols_config.system_prompt_path` is configured, use the
       prompt loaded from that file.
    3. Else use the mode's default prompt (`QUERY_SYSTEM_INSTRUCTION` for
       ASK, `TROUBLESHOOTING_SYSTEM_INSTRUCTION` for TROUBLESHOOTING).

## Configuration Surface

| Field Path | Type | Default | Purpose |
|---|---|---|---|
| `LLMRequest.mode` | enum | `ask` | Per-request mode selection (`ask` or `troubleshooting`) |
| `ols_config.max_iterations` | int | None (mode default) | Raise the tool-calling iteration cap above the mode default |
| `ols_config.system_prompt_path` | string | None | Override the mode-default system prompt with a custom file |
| `dev_config.enable_system_prompt_override` | bool | false | Allow per-request system prompt override (dev only) |

## Constraints

1. **Two modes only.** The `QueryMode` enum defines exactly two values:
   `ask` and `troubleshooting`. No other mode values are accepted.

2. **Mode does not persist.** Mode is a per-request property, not stored in
   the conversation cache. Each request independently declares its mode.

3. **Config cannot lower mode defaults.** Setting `ols_config.max_iterations`
   to a value below the mode's built-in default has no effect; the mode
   default is used instead.

4. **Cluster version requires Kubernetes API.** TROUBLESHOOTING mode
   retrieves the cluster version from the Kubernetes API. If the cluster
   version is unavailable, a sentinel value is used and the prompt reflects
   that the version is unknown.

5. **Custom system prompt overrides both modes.** When
   `ols_config.system_prompt_path` is set, the same custom prompt is used
   for both ASK and TROUBLESHOOTING requests. Mode-specific agent
   instructions are still appended when tool calling is enabled.

6. **Granite agent instructions are ASK-only.** The Granite-specific agent
   instruction variant (`AGENT_INSTRUCTION_GRANITE`) is only used in ASK
   mode. TROUBLESHOOTING mode uses its own dedicated agent instructions
   regardless of model family.

## Planned Changes

| Jira Key | Summary |
|---|---|
| OLS-2754 | Propose plan for autonomous investigation in TROUBLESHOOTING mode |
| OLS-2894 | Autonomous, policy-driven AI agents for OpenShift (TP: OCP 5.0) |
| OLS-2898 | Raise `max_iterations` to 50 and validate orchestration loop early-exit behavior |
