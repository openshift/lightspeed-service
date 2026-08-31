# Spec health report

Last evaluated: 2026-08-31
Trigger: staleness + accuracy check (spec-first alignment pass)
Layout: software (.ai/spec/)

## Stale

1. **how/project-structure.md — module map missing shipped files.** The `ols/` code tree
   now contains several files with no module-map entry:
   - `src/llms/providers/bedrock.py` (registered as `bedrock` — the AWS Bedrock provider,
     shipped via OLS-1895) and `src/llms/providers/utils.py` (shared provider helpers, e.g.
     Vertex OAuth scopes).
   - `src/tools/offloaded_content.py` (offloads large tool outputs to disk with search+read
     retrieval; `cleanup_offload_storage()` is called at startup in `app/main.py`).
   - `utils/audit_logger.py` and `utils/otel.py` (OTel-based audit logging — implements
     `what/audit-logging.md`).
   - `src/rag/stop_words.py` and `src/rag_index/solr_support.py` (RAG helpers).

2. **how/project-structure.md — metrics list incomplete.** The `app/metrics/metrics.py` row
   lists only seven `ols_*` metrics. Code also defines `ols_llm_reasoning_token_total` and the
   OTel GenAI histograms `gen_ai_client_token_usage`, `gen_ai_client_operation_duration_seconds`,
   and `gen_ai_execute_tool_duration_seconds` (all already documented in `what/observability.md`).

3. **how/project-structure.md — middleware stack stale.** The Middleware section describes only
   two `@app.middleware` functions and names `rest_api_counter` as the outermost layer. Code adds
   a third, class-based ASGI middleware `_RequestBodyLimitMiddleware` registered via
   `app.add_middleware()` as the outermost layer (rejects bodies over
   `constants.MAX_REQUEST_BODY_SIZE` = 2 MiB with HTTP 413).

4. **how/project-structure.md — stale tool-calling Implementation Note.** The note "Tool calling
   uses a multi-round streaming loop" attributes `iterate_with_tools()` to `DocsSummarizer`. That
   loop moved to `LLMExecutionAgent` (`_iterate_with_tools()`, `_invoke_llm()`,
   `_process_tool_calls_for_round()`, `_collect_round_llm_chunks()`); `DocsSummarizer` delegates
   via `self._llm_agent.execute()`. (Module map and Data Flow were already updated; this note lagged.)

5. **OLS-3221 markers claim shipped, but the work is largely unshipped.** `what/api.md`,
   `what/conversation-history.md`, and `how/cache.md` mark PostgreSQL-resilience behavior with
   `[NEW: OLS-3221]` / `[CHANGED: OLS-3221]` (implying merged), while each file's own Planned
   Changes section still lists OLS-3221 as `[PLANNED]` — an internal contradiction. Code check:
   - Shipped: the `@connection` transparent-reconnect decorator and `connected()` liveness check
     in `utils/postgres.py`; per-instance `_tx_lock` in `postgres_cache.py`.
   - NOT shipped: background health-check loop, dual-feed health status, statement/lock timeouts,
     health-status-backed readiness/liveness probes (liveness still returns `alive=True`
     unconditionally; readiness calls `conversation_cache.ready()` directly), and the config
     fields `cache_health_check_interval`, `statement_timeout`, `lock_timeout`,
     `liveness_db_failure_threshold` (absent from `app/models/config.py`).
   The unshipped markers were changed to `[PLANNED: OLS-3221]` to match code and the convention
   in README (unimplemented behavior uses `[PLANNED]`).

6. **README.md — provider count.** The what/ index called `llm-providers.md` "8 providers"; there
   are now nine provider types (OpenAI, Azure OpenAI, WatsonX, RHOAI vLLM, RHEL AI vLLM, Google
   Vertex Gemini, Google Vertex Anthropic, AWS Bedrock, Fake).

## Missing

1. **what/api.md — request-body size limit not specced as a behavioral rule.** The Middleware
   section (rules 26–31) does not mention the 2 MiB request-body limit that returns HTTP 413.
   This is shipped, client-visible behavior. Left for human decision to avoid inventing a
   behavioral rule; the implementation detail was added to `how/project-structure.md`.

2. **Offloaded tool-output storage has no behavioral spec.** `src/tools/offloaded_content.py`
   (offload large tool outputs to disk, search+read retrieval, `ols_config.offload_storage_path`)
   is not described in `what/tools.md` or `how/tools.md`. Only the module-map pointer was added.
   Human should decide whether it warrants behavioral coverage.

## Structural concerns

1. **what/audit-logging.md self-referential "parent spec".** Lines 3 and 145 cite the parent spec
   as `ols/.ai/spec/what/audit-logging.md`, which resolves to this same file (and the `ols/` prefix
   does not exist in this repo). Likely intended to point at a workspace-level or cross-repo parent
   spec. Not edited — the intended target is unknown; flagged for human clarification.

2. **what/api.md is large (~789 lines).** Unchanged concern from prior reports; splitting by
   endpoint category is optional and only worthwhile if it keeps growing.

## Findability issues

None new. The README index and what/↔how/ cross-reference table remain comprehensive.

## No issues (verified current)

- Provider registry: nine `@register_llm_provider_as` types match `what/llm-providers.md`
  (bedrock included) and `constants.SUPPORTED_PROVIDER_TYPES`.
- Reasoning-token support (`[PLANNED: OLS-3442]`) is correctly still planned: no `reasoning_config`
  in provider code, no `ChatVLLMReasoning`; OpenAI still uses `reasoning_effort` model-name detection.
- `what/observability.md` metric names, labels, and GenAI histograms match `app/metrics/metrics.py`.
- `what/audit-logging.md` span/attribute model matches `utils/audit_logger.py` + `utils/otel.py`.
- All 16 API endpoints and routers match `what/api.md` and `how/project-structure.md`.
- Cache, config, quota, auth how/ specs match their implementations (aside from the OLS-3221
  markers noted above).
