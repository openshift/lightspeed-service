# OpenShift LightSpeed Service -- Specifications

These specs define the requirements, behaviors, and architecture for the OLS lightspeed-service. They are organized into two layers optimized for AI agent consumption.

## Structure

| Layer | Path | Purpose |
|---|---|---|
| **what/** | `.ai/spec/what/` | Behavioral rules. What the system must do. Implementation-agnostic. |
| **how/** | `.ai/spec/how/` | Codebase navigation. How the code is organized. Implementation-specific. |

### what/ -- Behavioral Specifications

| Spec | Description |
|------|-------------|
| [system-overview.md](what/system-overview.md) | Core capabilities, user personas, deployment models, system boundaries |
| [api.md](what/api.md) | REST API contracts: all 16 endpoints, request/response shapes, streaming events, middleware |
| [query-processing.md](what/query-processing.md) | 8-stage pipeline from user query to LLM response: redaction, RAG, history, skills, tools, storage |
| [agent-modes.md](what/agent-modes.md) | ASK vs TROUBLESHOOTING: iteration limits, system prompts, behavioral differences |
| [conversation-history.md](what/conversation-history.md) | Storage backends, CRUD operations, compression, user isolation, concurrency |
| [llm-providers.md](what/llm-providers.md) | Provider contract, 8 providers with deviations, parameter system, reasoning models |
| [rag.md](what/rag.md) | Document retrieval via FAISS, multi-index, BYOK, hybrid RAG (tool/skill filtering only) |
| [auth.md](what/auth.md) | Three auth modules (k8s, noop, noop-with-token), permission scopes, user identity |
| [tools.md](what/tools.md) | MCP tool integration: gathering, execution, token budget, filtering, approval workflow |
| [skills.md](what/skills.md) | Skill discovery, selection via hybrid RAG, file concatenation, prompt injection |
| [quota.md](what/quota.md) | Token usage limits per user and cluster, scheduler, usage history |
| [config.md](what/config.md) | Configuration system: loading, validation, dynamic reload, all config sections |
| [security.md](what/security.md) | TLS, FIPS, PII redaction, credential handling, tool approval security, MCP credentials |
| [observability.md](what/observability.md) | Prometheus metrics (exact names), transcripts, feedback, logging, profiling |
| [prompts.md](what/prompts.md) | System prompts (full text), agent instructions, composition order, placeholders |
| [mcp-apps.md](what/mcp-apps.md) | MCP apps UI integration: resource discovery, direct tool calls, client auth headers |

### how/ -- Architecture Specifications

| Spec | Description |
|------|-------------|
| [project-structure.md](how/project-structure.md) | Directory layout, module responsibilities, startup sequence, layer integration |
| [query-pipeline.md](how/query-pipeline.md) | DocsSummarizer orchestration, stage-by-stage data flow, token budget algorithm, streaming interleaving |
| [llm-providers.md](how/llm-providers.md) | Provider registry, decorator discovery, class hierarchy, parameter mapping, HTTP client setup |
| [tools.md](how/tools.md) | MCP client lifecycle, tool gathering pipeline, approval state machine, retry/backoff, parallel execution |
| [config.md](how/config.md) | AppConfig singleton, Pydantic class hierarchy, two-phase validation, dynamic reload mechanism |
| [cache.md](how/cache.md) | Abstract cache interface, PostgreSQL schema, advisory locks, JSON serialization, in-memory LRU |

## Scope

These specs cover the **lightspeed-service** Python application only. The operator, console plugin, and RAG content pipeline are separate projects.

## Audience

AI agents. Content is optimized for precision and machine consumption.

## Quick Start

| I want to... | Read |
|--------------|------|
| Understand what OLS does | `what/system-overview.md` |
| Fix a bug in query processing | `what/query-processing.md` + `how/query-pipeline.md` |
| Add a new LLM provider | `what/llm-providers.md` + `how/llm-providers.md` |
| Understand the API | `what/api.md` |
| Navigate the codebase | `how/project-structure.md` |
| See what's planned | Look for `[PLANNED: OLS-XXXX]` in `what/` specs |

## Cross-Reference

When what/ and how/ file names don't match 1:1, this table maps behavioral specs to their implementation guides:

| what/ | how/ |
|---|---|
| `system-overview.md` | `project-structure.md` |
| `query-processing.md` | `query-pipeline.md` |
| `conversation-history.md` | `cache.md` |
| `config.md` | `config.md` |
| `llm-providers.md` | `llm-providers.md` |
| `tools.md` | `tools.md` |
| `api.md` | _(no dedicated how/ -- see `project-structure.md` for router/endpoint layout)_ |
| `agent-modes.md` | _(behavioral only -- mode selection in `how/query-pipeline.md`)_ |
| `auth.md` | _(see `how/project-structure.md` auth section)_ |
| `quota.md` | _(see `how/project-structure.md` quota section)_ |
| `rag.md` | _(see `how/project-structure.md` RAG section)_ |
| `skills.md` | _(see `how/project-structure.md` skills section)_ |
| `security.md` | _(cross-cutting -- spans multiple how/ files)_ |
| `observability.md` | _(see `how/project-structure.md` metrics section)_ |
| `prompts.md` | _(see `how/query-pipeline.md` prompt generation)_ |
| `mcp-apps.md` | _(see `how/tools.md` MCP client section)_ |

## Project History

OpenShift Lightspeed evolved through these phases:

1. **Prototype** (Q4 2023): Basic chat with OpenAI + RAG from product docs
2. **Early Access** (Q1-Q2 2024): Configuration system, conversation cache, auth, multi-provider, metrics, feedback
3. **Tech Preview** (Q3 2024): Streaming, RHOAI/RHELAI providers, Azure Entra ID, TLS profiles
4. **GA** (Q4 2024 - Q1 2025): PostgreSQL cache, quota management, FIPS, disconnected mode, transcripts
5. **Post-GA** (2025-2026): MCP tools, context-aware queries, skills, tool approval, BYOK, custom system prompts, query modes, dynamic config reload

## Conventions

- **Rule numbering:** behavioral rules are numbered sequentially within each what/ file.
- **Planned changes:** unimplemented behavior is marked with `[PLANNED]` or `[PLANNED: OLS-XXXX]` inline next to the rule it affects.
- **Constraints:** component-specific and cross-cutting constraints go in the relevant what/ file's Constraints section, co-located with behavioral rules. Development conventions go in CLAUDE.md.
- **Authority:** what/ specs are authoritative for behavior. how/ specs are authoritative for implementation. When they conflict, what/ wins.
- **When to create a new file vs. extend an existing one:** if the new concern has its own lifecycle, configuration surface, and can be understood independently, it gets its own file. If it's a capability added to an existing component, it goes in that component's file.
- **Config field names** reference `olsconfig.yaml` paths (e.g., `ols_config.tool_filtering.threshold`).
- **Internal constants** are stated as behavioral rules without numeric values; `how/` specs may include specific values.
