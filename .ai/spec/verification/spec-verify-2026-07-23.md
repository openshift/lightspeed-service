# Verification Report: lightspeed-service Spec
Verified: 2026-07-23
Spec root: /Users/xavi/street/github.com/AI/ols/lightspeed-service/.ai/spec/

## Summary
- 5 broken or inaccurate internal references
- 4 internal inconsistencies
- 3 completeness gaps
- 2 cross-repo alignment issues

## Reference Issues

**REF-1. `audit-logging.md` missing from README spec index.** (Medium)
`what/audit-logging.md` exists with 13 behavioral rules but is not listed in README's what/ spec table (lists 16 files; audit-logging.md is the 17th) or the cross-reference table (lines 68-97).

**REF-2. Duplicate Bedrock sections in `llm-providers.md`.** (High)
Two complete `### AWS Bedrock (bedrock)` section headers at lines 127 and 138, each with independently numbered Rules 42-46 but different content. Only one section should exist; the second (lines 138-151) is more complete and should be retained.

**REF-3. Duplicate Rule 3 in `conversation-history.md`.** (Medium)
Rule 3 appears twice: once as `[PLANNED: OLS-3442]` about storing reasoning content, and once as the rule about retrieving history. The second Rule 3 and all subsequent rules are off by one.

**REF-4. Fake Provider rules misnumbered in `llm-providers.md`.** (Low)
Fake Provider section uses Rules 39-41, numerically preceding Bedrock rules 42-46. Consequence of the duplicate Bedrock insertion. Should be renumbered to follow the last Bedrock rule after deduplication.

**REF-5. `health-report.md` is entirely stale.** (Medium)
All issues flagged in the 2026-05-29 health report have been resolved (how/auth.md, how/quota.md, LLMExecutionAgent, OLS-2521). The file should be regenerated or removed.

## Internal Inconsistencies

**INC-1. Tool approval config field name conflict.** (High)
`what/tools.md` lines 143, 209 use `tools_approval.strategy`. `what/security.md` line 104 and `what/system-overview.md` line 98 use `ols_config.tools_approval.approval_type`. Same field, two names. Both refer to the `never`/`always`/`tool_annotations` selector.

**INC-2. `system-overview.md` Rule 7 describes only BYOK RAG path.** (Low)
Rule 7: "Before sending a question to the LLM, the service retrieves relevant chunks from a pre-built product documentation index." Accurate for BYOK/FAISS but not for OKP, where retrieval happens during LLM tool-calling (Stage 7) via the `search_openshift_documentation` tool. Parent system-overview correctly describes both paths.

**INC-3. Duplicate Bedrock section content divergence.** (High)
Beyond being duplicated (REF-2), the two Bedrock sections contain different information. First section covers Mantle gateway URL and two auth modes (condensed). Second section adds: no-URL rejection rule, `httpx-aws-auth` SigV4 signing detail, `BedrockParameters` union, certificate/TLS details. The second section should be the sole version.

**INC-4. `system-overview.md` marks Bedrock as PLANNED but it is shipped.** (Medium)
`system-overview.md` line 77 says `[PLANNED: OLS-1680]` and lists OLS-1680 in Planned Changes. But `what/llm-providers.md` fully documents Bedrock as implemented, with its own Planned Changes showing OLS-1680 as struck through ("Implemented in OLS-1895."). The PLANNED marker and entry should be removed from system-overview.md.

## Completeness Gaps

**GAP-1. `audit-logging.md` not indexed or cross-referenced.** (Medium)
As noted in REF-1. The file should be added to the README's what/ spec table and cross-reference table.

**GAP-2. No how/ spec for OKP/Solr retrieval.** (Low)
The OKP retrieval path is documented in `what/rag.md` Rules 1-8 but has no corresponding how/ architecture spec. BYOK is indirectly covered by the query pipeline how/ spec.

**GAP-3. Stale health report.** (Low)
As noted in REF-5. The 2026-05-29 health report lists only resolved issues and provides no current value. Should be regenerated or removed.

## Cross-Repo Alignment Issues

**ALIGN-1. Parent spec lists OLS-2521 and OLS-2776 as planned; service specs show them shipped.** (Medium)
Parent `query-pipeline.md` Planned Changes table lists:
- `OLS-2521 | Support Google Gemini as direct LLM provider`
- `OLS-2776 | Support Anthropic as direct LLM provider`

Both are fully implemented in lightspeed-service (`google_vertex` and `google_vertex_anthropic` provider types, complete rules). Planned Changes entries should be struck through or removed in the parent.

**ALIGN-2. Parent spec lists OLS-3442 as fully planned; service specs show partial implementation.** (Low)
Parent `query-pipeline.md` line 149 lists OLS-3442 (reasoning token support) as planned. Service specs show partial implementation: `reasoning_config` is documented as working for most providers, streaming accumulation is documented, `ChatVLLMReasoning` is in `what/query-processing.md` Rule 50. Some PLANNED markers remain within the service specs (e.g., `conversation-history.md` Rule 3). Parent spec should say "partially implemented" rather than fully planned.

## Files Checked

### what/ (17 files)
| File | Rules | Status |
|---|---|---|
| system-overview.md | 42 rules + 11 constraints | INC-2, INC-4 |
| api.md | 31 rules + 12 constraints | Clean |
| query-processing.md | 54 rules + 7 constraints | Clean |
| agent-modes.md | 20 rules + 4 constraints | Clean |
| conversation-history.md | 23 rules + 4 constraints | REF-3 |
| llm-providers.md | 46 rules (with duplicates) + 3 constraints | REF-2, REF-4, INC-3 |
| rag.md | 22 rules + 6 constraints | Clean |
| auth.md | 47 rules + 6 constraints | Clean |
| tools.md | 35 rules + 6 constraints | INC-1 |
| skills.md | 22 rules + 4 constraints | Clean |
| quota.md | 13 rules + 4 constraints | Clean |
| config.md | 20 rules + 6 constraints | Clean |
| security.md | 37 rules + 4 constraints | INC-1 |
| observability.md | 27 rules + 6 constraints | Clean |
| prompts.md | 15 rules + 2 constraints | Clean |
| mcp-apps.md | 12 rules + 4 constraints | Clean |
| audit-logging.md | 13 rules + 5 constraints | REF-1, GAP-1 |

### how/ (9 files)
- project-structure.md, query-pipeline.md, auth.md, cache.md, config.md, llm-providers.md, tools.md, quota.md, e2e-bedrock.md — all clean

### Other
- README.md — REF-1, GAP-1
- decisions/README.md — clean placeholder
- health-report.md — REF-5, GAP-3

### Parent specs checked
- ols/.ai/spec/what/query-pipeline.md — ALIGN-1, ALIGN-2
- ols/.ai/spec/what/system-overview.md — clean
- ols/.ai/spec/what/audit-logging.md — exists (confirms parent ref is valid)
