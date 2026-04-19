# Behavioral Specifications (what/)

These specs define WHAT the OLS service must do -- testable behavioral rules, configuration surface, constraints, and planned changes. They are technology-neutral where possible and survive a complete rewrite in a different framework.

## Spec Index

| Spec | Description |
|------|-------------|
| [system-overview.md](system-overview.md) | Core capabilities, user personas, deployment models, system boundaries |
| [api.md](api.md) | REST API contracts: all 16 endpoints, request/response shapes, streaming events, middleware |
| [query-processing.md](query-processing.md) | 8-stage pipeline from user query to LLM response: redaction, RAG, history, skills, tools, storage |
| [agent-modes.md](agent-modes.md) | ASK vs TROUBLESHOOTING: iteration limits, system prompts, behavioral differences |
| [conversation-history.md](conversation-history.md) | Storage backends, CRUD operations, compression, user isolation, concurrency |
| [llm-providers.md](llm-providers.md) | Provider contract, 8 providers with deviations, parameter system, reasoning models |
| [rag.md](rag.md) | Document retrieval via FAISS, multi-index, BYOK, hybrid RAG (tool/skill filtering only) |
| [auth.md](auth.md) | Three auth modules (k8s, noop, noop-with-token), permission scopes, user identity |
| [tools.md](tools.md) | MCP tool integration: gathering, execution, token budget, filtering, approval workflow |
| [skills.md](skills.md) | Skill discovery, selection via hybrid RAG, file concatenation, prompt injection |
| [quota.md](quota.md) | Token usage limits per user and cluster, scheduler, usage history |
| [config.md](config.md) | Configuration system: loading, validation, dynamic reload, all config sections |
| [security.md](security.md) | TLS, FIPS, PII redaction, credential handling, tool approval security, MCP credentials |
| [observability.md](observability.md) | Prometheus metrics (exact names), transcripts, feedback, logging, profiling |
| [prompts.md](prompts.md) | System prompts (full text), agent instructions, composition order, placeholders |
| [mcp-apps.md](mcp-apps.md) | MCP apps UI integration: resource discovery, direct tool calls, client auth headers |

## How to Use These Specs

- **Fixing a bug**: Read the relevant spec to understand correct behavior, then compare against the code.
- **Adding a feature**: Check if the spec covers the requirement. Update the spec before implementing.
- **Refactoring**: Use the specs as acceptance criteria. The implementation can change freely as long as it meets the behavioral rules.
- **Understanding planned work**: Look for `[PLANNED: OLS-XXXX]` markers inline and "Planned Changes" sections.

## Relationship to how/ Specs

These `what/` specs define the behavioral contract. The [`how/` specs](../how/README.md) describe the current implementation architecture. Read `what/` to understand requirements, read `how/` to understand the codebase structure.
