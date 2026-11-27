# Chat on the Shared Execution Core

[PLANNED: OLS-4671] Not implemented. The service stops running its own tool-calling loop and runs Stage 7 of the query pipeline on the shared DeepAgents execution core library (`lightspeed-agent-core`, placeholder name). Cross-repo contract (authoritative for integration and shared behavior): workspace `.ai/spec/what/unified-execution-core.md`; decision: workspace `.ai/spec/decisions/0052-shared-deepagents-execution-core.md`.

This file defines the service-side rules. Where it conflicts with existing rules in other service specs, this file wins for the post-cutover state, and those rules carry `[PLANNED: OLS-4671]` markers pointing here.

## Behavioral Rules

### Responsibilities

1. After cutover the service MUST NOT contain an agent loop. Stage 7 (LLM generation with tool calling) MUST call the core. The service keeps:
   - stages 1–2 (validation, redaction, attachments)
   - retrieval: BYOK RAG chunks (top-k, score cutoff, multi-index merge, URL deduplication), skill matching, tool filtering
   - prompt composition
   - quota, the conversation endpoints and conversation record (rules 6b–6c), transcripts and feedback
   - the REST/SSE API
   - the pending-decision store and the decision endpoint
2. The service MUST build the core agent with the chat profile:
   - Tools: the MCP tools it resolved and filtered for the request, plus `search_openshift_documentation` when OKP/Solr is configured.
   - Backend: an in-memory request-scoped backend. No `execute`, `write_file` or `edit_file`; no subagents; no DeepAgents skills middleware.
   - Prompt and messages: the composed system prompt, plus the retrieved RAG context and matched skill content, passed through unchanged. Prior conversation messages come from the conversation's checkpoint (rule 6a), not from the service passing stored history.
3. The service MUST pass the model's context window (`context_window_size`) to the core for models without a LangChain model profile, so DeepAgents' default thresholds can be computed. No other context-management settings are passed.
4. Provider construction moves to the core's model router. The service maps each configured `llm_providers[]` entry to a core route. The `olsconfig.yaml` provider schema does not change, and every currently supported provider type MUST map to a route.

### Removed behavior

5. The service MUST remove:
   - prompt-budget computation and truncation of RAG chunks to a budget
   - the `TokenBudgetTracker`
   - history truncation and LLM history compression
   - the prompt-length check before the provider call (`PromptTooLongError` raised by the token handler)
   - the tool budget, per-round cap and tiered truncation
   - the large-output offload (`offloaded_content.py`, `cleanup_offload_storage()`)
   - the Classic tool-calling loop (`DocsSummarizer` / `LLMExecutionAgent` orchestration)
6. Context management is delegated to the core (DeepAgents eviction and summarization with default thresholds), within a turn and across turns.
6a. **Conversation state.** Each `conversation_id` MUST be run as a DeepAgents thread with a LangGraph checkpointer supplied by the service: a Postgres checkpointer in the existing OLS database when `conversation_cache` is PostgreSQL, an in-memory checkpointer when it is memory. Summaries persist across turns, so compaction is not repeated on every turn. Only the latest checkpoint per thread is kept, written when a run completes. A run that does not complete (disconnect, error, inspection failure, quota error) writes no checkpoint.
6b. **Conversation record.** `GET`/`PUT`/`DELETE /v1/conversations...`, user isolation and capacity eviction keep their contracts, and each completed turn is still recorded as today (query and plain-string response, including reasoning). Deleting or evicting a conversation MUST also delete its checkpoint thread. Whether the endpoints keep reading the existing cache as a projection (recommended) or read from the checkpoint is workspace open decision OD-1.
6c. **Provider or model change.** Handling a conversation whose effective provider or model changes between turns is workspace open decision OD-2 (recommended: start a new thread seeded from the plain-string history).
6d. **Data at rest.** Checkpoints store tool calls and redacted, inspection-passed tool results, which the conversation cache does not store today. A security and data-retention review MUST be recorded before cutover.
7. **Config compatibility:** `ols_config.history_compression_enabled`, `ols_config.tool_round_cap_fraction`, `ols_config.offload_storage_path` and `llm_providers[].models[].parameters.tool_budget_ratio` MUST still be accepted and validated, so existing configurations keep loading. They have no effect after cutover and MUST log a one-time deprecation warning when set.

### Streaming

8. The service MUST map core events to the existing SSE contract (`what/api.md`) without changing event names, payloads, plain-text formats, or ordering rules.
9. DeepAgents summarization start and end MUST be emitted as `history_compression_start` and `history_compression_end`.
10. `skill_selected` stays a service event from skill matching.
11. Streamed reasoning continues to be accumulated into the stored plain-string response.

### Context overflow

12. When the core raises `ContextWindowExceeded` (DeepAgents' reactive compaction could not recover), the service MUST return today's "Prompt is too long" response: HTTP 413 for `/v1/query`, and the `error` event for streaming, with the same body and plain-text form.
12a. Input that summarization cannot shrink (system prompt, RAG context, skill content, attachments, current query) is not trimmed or rejected locally, and RAG chunks are not trimmed to a token budget. Oversized input reaches the provider and fails with rule 12's response. This is an accepted behavior change measured by the parity gate.

### Tool execution

13. The service MUST enable the core's tool retries with the current policy (`what/tools.md` rules 11–13).
14. Inspection, wrapping and redaction use the core (sandbox implementation):
    - Each result is inspected individually.
    - Any failure stops the run, emits the fixed inspection-failure `error` event and ends the stream (HTTP 500 for `/v1/query`).
    - The all-or-nothing concurrent-round rule (`what/tools.md` rule 18c) is superseded.
    - Rejected content MUST NOT enter SSE, history, or transcripts. A completed native tool span may keep the raw (sanitized) result even if inspection later rejects it. This requires compliance sign-off before cutover.
15. **Run timeout:** the service passes `TOOL_CALL_ROUND_TIMEOUT` multiplied by the effective turn limit to the core's cooperative run timeout, restarted on each Continue. The clock pauses while awaiting approval or continuation decisions. When the timeout is reached, the service returns today's user-facing timeout message.

### Approval

16. The service MUST supply the core's `ApprovalMiddleware` with an approver implementing today's approval contract (`what/tools.md` rules 27–31):
    - `approval_required`
    - `POST /v1/tool-approvals/decision` with `approval_id`
    - 404/409 semantics
    - `tools_approval.approval_timeout`
    - pending-state cleanup

    The approval strategy (`never`, `always`, `tool_annotations`) is evaluated per call. LangGraph human-in-the-loop interrupts and checkpointers MUST NOT be used.

### Turn limit and continuation

17. The service MUST supply the core's `TurnLimitMiddleware` with the effective turn limit (`what/agent-modes.md` rule 12). Turns are model calls. A run whose model returns a complete answer within the limit ends normally, with no continuation decision.
18. When the limit is reached and the model is still requesting tools:
    - The service MUST register a pending continuation and emit `continuation_required` with `continuation_id`, `turns_used` and `turn_limit`.
    - The wait is bounded by `tools_approval.approval_timeout`. On expiry the run takes the Stop path (rule 21).
    - Client disconnect cancels the request and removes the pending entry.
18a. **Pauses are single-stream.** Approval and continuation waits exist only inside the live request. They do not survive a disconnect or a pod restart and cannot be resumed later. The checkpointer MUST NOT be used to suspend or resume a run.
18b. **Non-streaming.** `/v1/query` cannot prompt the user. At the turn limit it MUST take the Stop path immediately, without registering a continuation.
19. The decision arrives on `POST /v1/tool-approvals/decision` with `continuation_id` and `approved`: `true` means Continue, `false` means Stop.
20. **Continue** grants another block of turns equal to the effective limit, in the same stream.
21. **Stop** triggers one final model call with tool calling disabled, using the existing per-provider mechanisms (`what/agent-modes.md` rule 13), and an instruction to answer from the information gathered so far and state what is incomplete.
    - If tool calls are still returned, they are ignored and the stream ends with a fixed turn-limit message.
    - The final answer streams as `token`/`reasoning` events followed by `end`.

### Usage and observability

22. Quota MUST be deducted from the core's usage for all model calls of the request: agent turns, the Stop final answer, summarization, and inspection classifier calls.
22a. **Quota enforcement.** The existing soft cap (`what/quota.md` rules 3–4e) applies to every model call: the service supplies a quota gate that the core calls before each model call, and debits after each call rather than once per request. When the gate reports exhausted quota, the run stops with the existing quota error. Continue grants turns, not quota; with quota exhausted, the Stop final answer is skipped and the quota error is returned.
23. GenAI spans come from the core and are exported only when OTLP export is configured. Prometheus metrics, transcripts and feedback stay in the service.

### Dependency

24. The service MUST consume the core as a hash-pinned HTTPS direct-URL requirement to the GitHub release sdist, with only the provider extras the service supports. It MUST add the core's build backend to `.konflux/requirements-build.txt`.

## Constraints

1. No `olsconfig.yaml`, OLSConfig, or operator change is required. The only API additions are `continuation_required` and `continuation_id`. Checkpoint tables are added to the existing OLS PostgreSQL database.
2. Cutover is a single release with no feature flag. lightspeed-console ships the continuation prompt in the same release.
3. Cutover is blocked until the workspace parity gate passes (workspace `unified-execution-core.md` rules 39–42).

## Planned Changes

| Ticket | Summary |
|---|---|
| OLS-4671 | Run Stage 7 on the shared DeepAgents execution core; remove budget, compression, offload and the Classic loop; add turn-limit continuation |
