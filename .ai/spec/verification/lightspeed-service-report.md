# Spec Verification Report: lightspeed-service

**Date:** 2026-07-24
**Verifier:** Independent spec verification agent
**Scope:** All `what/` spec files in `lightspeed-service/.ai/spec/what/`

---

## Pass 1: Acceptance Criteria

Searched all what/ files for `- [ ]` acceptance criteria checkboxes.

**Result:** No `- [ ]` acceptance criteria checkboxes found in any what/ file. The specs use numbered behavioral rules rather than checkbox-style acceptance criteria.

**Verdict:** N/A — no criteria to evaluate.

---

## Pass 2: Constraint Compliance

Checked all what/ files against the 8 cross-repo constraints in `ols/.ai/spec/constraints.md`.

### Constraint 1: Fork-based workflow
Not applicable to spec content (process constraint, not behavioral spec).
**PASS**

### Constraint 2: Commit messages start with OLS-XXXX
Not applicable to spec content (process constraint).
**PASS**

### Constraint 3: Squash commits
Not applicable to spec content (process constraint).
**PASS**

### Constraint 4: Jira project key is OLS on redhat.atlassian.net
All Jira references in specs use OLS-XXXX format.
**PASS**

### Constraint 5: Classic OLS CRDs use API group ols.openshift.io/v1alpha1
No CRD API group references found in the what/ specs — CRDs are in the operator repo.
**PASS** (not applicable)

### Constraint 6: Agentic OLS CRDs use API group agentic.openshift.io/v1alpha1
No agentic CRD references in lightspeed-service specs.
**PASS** (not applicable)

### Constraint 7: All components deploy into openshift-lightspeed namespace
No namespace references in what/ specs — deployment is an operator concern.
**PASS** (not applicable)

### Constraint 8: Embedding model for building RAG indexes must match the model used to query them
- `what/rag.md` Constraint 2: "The BYOK embedding model used for retrieval must be the same model used to create the index. A mismatch will produce meaningless similarity scores." — Directly restates the cross-repo constraint.
- `what/skills.md` Constraint 3: "The embedding model used for skill retrieval at query time must be the same model used to populate the index at startup."
**PASS**

### Summary: 0 violations out of 8 constraints.

---

## Pass 3: Term Consistency

**Skipped** — no glossary file exists.

---

## Pass 4: Internal Reference Accuracy

### References from what/audit-logging.md

| Reference | Target | Exists? | Accurate? |
|---|---|---|---|
| `observability.md` | what/observability.md | YES | PASS — covers Prometheus metrics and gen_ai.* histogram metrics |
| `query-processing.md` | what/query-processing.md | YES | PASS — covers pipeline stages where spans would be created |
| `tools.md` | what/tools.md | YES | PASS — covers tool execution flow, MCP integration |
| `auth.md` | what/auth.md | YES | PASS — covers k8s token validation and user identity extraction |
| `ols/.ai/spec/what/audit-logging.md` (parent spec) | workspace-level spec | YES | PASS — file exists at `ols/.ai/spec/what/audit-logging.md` |

### References from what/api.md

| Reference | Target | Exists? | Accurate? |
|---|---|---|---|
| `what/query-processing.md` (Rule 9) | what/query-processing.md | YES | PASS |
| `what/tools.md` (Config Surface) | what/tools.md | YES | PASS |
| `what/quota.md` (Config Surface) | what/quota.md | YES | PASS |

### References from what/config.md

| Reference | Target | Exists? | Accurate? |
|---|---|---|---|
| `what/llm-providers.md` (table) | what/llm-providers.md | YES | PASS |
| `what/tools.md` (table) | what/tools.md | YES | PASS |
| `what/auth.md` (table) | what/auth.md | YES | PASS |
| `what/conversation-history.md` (table) | what/conversation-history.md | YES | PASS |
| `what/security.md` (table) | what/security.md | YES | PASS |
| `what/observability.md` (table) | what/observability.md | YES | PASS |
| `what/skills.md` (table) | what/skills.md | YES | PASS |
| `what/quota.md` (table) | what/quota.md | YES | PASS |
| `what/rag.md` (table) | what/rag.md | YES | PASS |

### References from what/conversation-history.md

No cross-file references found (self-contained).
**PASS**

### References from what/mcp-apps.md

| Reference | Target | Exists? | Accurate? |
|---|---|---|---|
| `what/tools.md` (Constraint 1, Rule 18, Config Surface) | what/tools.md | YES | PASS — covers tool execution rules, header resolution (rules 2-3), and MCP server config |

### References from what/agent-modes.md

No cross-file references found (self-contained; references are to constants/code, not other spec files).
**PASS**

### References from what/query-processing.md

| Reference | Target | Exists? | Accurate? |
|---|---|---|---|
| `what/rag.md` (Rule 14) | what/rag.md | YES | PASS |
| `what/conversation-history.md` (Rules 18, 37) | what/conversation-history.md | YES | PASS |
| `what/agent-modes.md` (Rules 32, Config Surface) | what/agent-modes.md | YES | PASS |
| `what/tools.md` (Rule 32) | what/tools.md | YES | PASS |
| `what/skills.md` (Rule 23) | what/skills.md | YES | PASS |

### References from what/skills.md

| Reference | Target | Exists? | Accurate? |
|---|---|---|---|
| `what/query-processing.md` (Rules 9, 20) | what/query-processing.md | YES | PASS — Stage 5 covers skill selection, Stage 6 covers prompt injection |

### References from what/tools.md

| Reference | Target | Exists? | Accurate? |
|---|---|---|---|
| `what/query-processing.md` (Rules 10, 15) | what/query-processing.md | YES | PASS — Stage 7 covers tool calling loop, token budget system covers tool reserve |
| `what/agent-modes.md` (Rule 14) | what/agent-modes.md | YES | PASS — covers iteration limits |
| `what/rag.md` (Rule 19) | what/rag.md | YES | PASS — covers hybrid RAG mechanism |

### References from what/prompts.md

| Reference | Target | Exists? | Accurate? |
|---|---|---|---|
| `what/agent-modes.md` (Rules 1, 4) | what/agent-modes.md | YES | PASS |
| `what/query-processing.md` (implied by Rule 14 "Stage 6") | what/query-processing.md | YES | PASS |

### References from what/rag.md

No cross-references to other what/ files found (references are to config paths).
**PASS**

### References from what/security.md

No cross-references to other what/ or how/ files found.
**PASS**

### References from what/quota.md

No cross-references to other what/ files found.
**PASS**

### References from what/observability.md

No cross-references to other what/ files found.
**PASS**

### References from what/llm-providers.md

No cross-references to other what/ files found.
**PASS**

### References from what/system-overview.md

No cross-references to other what/ files found (references are all to config paths).
**PASS**

### Summary: 0 reference issues found.

---

## Cross-File Consistency Issues Found

### Issue 1: Inconsistent default timeout for MCP servers

- `what/tools.md` Configuration Surface table states `mcp_servers.servers[].timeout` default is **5** seconds.
- `what/mcp-apps.md` Configuration Surface table states `mcp_servers.servers[].timeout` default is **30** seconds (with note "30 (apps)").
- `what/mcp-apps.md` Rule 21 states "When no timeout is configured for a server, the default is 30 seconds."

**FINDING:** The default MCP server timeout differs between the tool execution path (5s in tools.md) and the MCP Apps path (30s in mcp-apps.md). If this is intentional (different defaults for different code paths), it should be made explicit. If the MCP server timeout is a single configured value, the specs contradict each other.

### Issue 2: OLS-3221 marked both [PLANNED] and [CHANGED]

- `what/conversation-history.md` Planned Changes section lists OLS-3221 as `[PLANNED: OLS-3221]`.
- `what/api.md` Rules 23-24 mark OLS-3221 as `[CHANGED: OLS-3221]`.
- `what/api.md` Config Surface lists fields as `[NEW: OLS-3221]`.
- `what/conversation-history.md` Rules 20-24 are marked `[NEW: OLS-3221]`.

**FINDING:** The conversation-history Planned Changes section still lists OLS-3221 as planned, but the rules themselves are already written as current behavior (marked [NEW]). The Planned Changes entry should be moved to a "Recently Completed" section or removed, since the spec content has already been written.

### Issue 3: Duplicate rule number in rag.md

- `what/rag.md` has BYOK rules numbered 1-13, then jumps to rule 15 for "Tool & Skill Filtering (Hybrid RAG)" section. Rule 14 is missing.

**FINDING:** Rule numbering gap — rule 14 is skipped in rag.md. The "Tool & Skill Filtering" section starts at rule 15 instead of 14.

### Issue 4: Duplicate rule number 14/14a in audit-logging.md

- `what/audit-logging.md` has rules 13, 14, 14a, 15. Rule 14a is a non-standard numbering convention that breaks the sequential numbering rule stated in README.md ("behavioral rules are numbered sequentially within each what/ file").

**FINDING:** Rule 14a in audit-logging.md should be renumbered to 15, with subsequent rules incremented.

### Issue 5: proxy_config field path inconsistency

- `what/security.md` Configuration Surface lists proxy config under `llm_providers[].proxy_config.proxy_url` (per-provider).
- `what/system-overview.md` references `ols_config.proxy_config` (global).
- `what/llm-providers.md` Rule 15 references `ols_config.proxy_config.proxy_url` (global).

**FINDING:** The proxy config field path in security.md (`llm_providers[].proxy_config`) differs from all other specs which use `ols_config.proxy_config`. The security.md reference appears incorrect — proxy config is a global setting, not per-provider.

---

## Summary

| Category | Result |
|---|---|
| Acceptance criteria (Pass 1) | N/A — no `- [ ]` criteria found |
| Constraint violations (Pass 2) | **0** violations |
| Term consistency (Pass 3) | Skipped (no glossary) |
| Reference issues (Pass 4) | **0** broken references |
| Cross-file consistency issues | **5** issues found |
