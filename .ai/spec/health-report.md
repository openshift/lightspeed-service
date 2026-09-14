# Spec health report

Last evaluated: 2026-08-31
Trigger: staleness + accuracy check (spec-first alignment pass)
Layout: software (.ai/spec/)

## Stale

None remaining. All staleness findings from the August 2026 audit were resolved
in this same alignment pass (see Resolved below).

## Resolved (August 2026 alignment pass)

1. **how/project-structure.md — module map missing shipped files.** Added entries for
   `src/llms/providers/bedrock.py`, `src/llms/providers/utils.py`,
   `src/tools/offloaded_content.py`, `utils/audit_logger.py`, `utils/otel.py`,
   `src/rag/stop_words.py`, `src/rag_index/solr_support.py`.

2. **how/project-structure.md — metrics list incomplete.** Added `ols_llm_reasoning_token_total`
   and OTel GenAI histograms to the metrics inventory.

3. **how/project-structure.md — middleware stack stale.** Added `_RequestBodyLimitMiddleware`
   as outermost ASGI middleware (2 MiB body limit, HTTP 413).

4. **how/project-structure.md — stale tool-calling Implementation Note.** Updated to attribute
   the multi-round tool loop to `LLMExecutionAgent`, not `DocsSummarizer`.

5. **OLS-3221 markers contradicted code.** Specs marked unshipped PostgreSQL-resilience behavior
   with `[NEW: OLS-3221]` / `[CHANGED: OLS-3221]` (implying merged). Corrected: shipped behavior
   (`@connection` pre-check reconnect, `connected()`, `_tx_lock`) left unmarked; unshipped behavior
   (error distinction, statement/lock timeouts, background health-check, dual-feed health status,
   health-backed probes) changed to `[PLANNED: OLS-3221]`.

6. **README.md — provider count.** Updated from "8 providers" to "9 provider types".

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
