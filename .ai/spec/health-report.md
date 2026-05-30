# Spec health report

Last evaluated: 2026-05-29
Trigger: structural evaluation (user-invoked)
Layout: software (.ai/spec/)

## Stale

1. **how/query-pipeline.md — tool-calling architecture refactored.** The spec describes `iterate_with_tools()`, `_invoke_llm()`, `_collect_round_llm_chunks()`, and `_process_tool_calls_for_round()` as methods of `DocsSummarizer`. They have been refactored into a separate `LLMExecutionAgent` class in `ols/src/query_helpers/llm_execution_agent.py`. `DocsSummarizer.generate_response()` now delegates to `self._llm_agent.execute()`.

2. **what/system-overview.md — Google Vertex providers shipped.** Rule 11 marks OLS-2521 (Google Gemini) and OLS-2776 (Anthropic) as `[PLANNED]`, but Google Vertex with both Gemini and Claude/Anthropic models is already merged and registered as `google_vertex` and `google_vertex_anthropic` provider types.

3. **what/llm-providers.md — same staleness.** Planned Changes section lists OLS-2521 as future work, but Gemini via Vertex is shipped.

## Missing

1. **how/auth.md** — Auth implementation spans 5 files and 549 LOC (K8s TokenReview, SubjectAccessReview, cluster ID retrieval, multiple auth strategies). Complex enough to warrant its own how/ file.

2. **how/quota.md** — Quota system spans 6 files and 420 LOC (multi-limiter coordination, PostgreSQL state, scheduler daemon thread, token usage history). No how/ file documents the implementation patterns.

3. **how/project-structure.md — missing LLMExecutionAgent.** The module map does not include `ols/src/query_helpers/llm_execution_agent.py`, which now contains the core tool-calling loop.

## Structural concerns

1. **what/api.md is 778 lines** — largest spec file by a wide margin. Covers all 16 endpoints in one file. Could be split by endpoint category (query, conversations, feedback/mcp, infrastructure) if it continues growing.

2. **what/observability.md minor boundary violation** — references `constants.py` by file name. What/ files should be implementation-agnostic; this should be generalized to "defined as a module constant."

## Findability issues

None. The cross-reference table added in the alignment pass maps all what/ to how/ files. The spec index in README.md is comprehensive.

## No issues

- All 8 provider files in code match the spec (openai, azure_openai, watsonx, rhoai_vllm, rhelai_vllm, google_vertex, fake_provider)
- All 16 API endpoints in code match what/api.md
- Skills system (`ols/src/skills/`) matches what/skills.md
- RAG system (`ols/src/rag/`, `ols/src/rag_index/`) matches what/rag.md
- Cache implementation matches how/cache.md
- Config implementation matches how/config.md
- LLM provider registry and loader match how/llm-providers.md
- Tool execution and approval match how/tools.md
- What/how separation is well-maintained across all files
- No unacceptable duplication between what/ and how/ layers
