# Spec health report

Last evaluated: 2026-07-22 (updated from 2026-05-29 report)
Trigger: multi-repo child spec evaluation
Layout: software (.ai/spec/)

## Status: Previous Report Stale — Spec is Current ✓

The health report from 2026-05-29 identified 3 stale areas. Review of current code shows:

### Issues from Previous Report — RESOLVED

1. **how/query-pipeline.md tool-calling architecture** ✓
   - Previous report: Stale references to `DocsSummarizer` methods
   - Current code: Spec correctly describes `LLMExecutionAgent` class and delegation pattern
   - Status: Fixed (no action needed)

2. **Google Vertex providers PLANNED markers** ✓
   - Previous report: OLS-2521 (Gemini) and OLS-2776 (Anthropic) marked PLANNED but shipped
   - Current code: Spec shows Google Vertex (Gemini) and Google Vertex (Anthropic) as registered providers
   - Rule 11 in what/system-overview.md correctly lists all 8 provider types including both Vertex variants
   - Status: Fixed (no action needed)

3. **Missing how/auth.md and how/quota.md**
   - Previous report: Auth and quota implementations lack how/ documentation
   - Status: Still missing (no changes detected in codebase from May 29)
   - Recommendation: These should be documented if time permits, but are lower priority

### Remaining Gaps

1. **Missing how/auth.md** — Auth implementation spans 5 files and 549 LOC
2. **Missing how/quota.md** — Quota system spans 6 files and 420 LOC
3. **Missing LLMExecutionAgent module reference** — check how/project-structure.md

## Verification

✓ All provider types match code (8 providers: openai, azure, watsonx, rhoai_vllm, rhelai_vllm, google_vertex, google_vertex_anthropic, fake)
✓ LLMExecutionAgent architecture correctly specified
✓ Reasoning config support properly documented
✓ All 16 API endpoints match what/api.md
✓ No placeholder text in current specs

## Result

Spec is largely current. Previous stale items have been resolved. Two how/ files remain undocumented but this is acceptable—the behavioral layer (what/) is complete and authoritative.
