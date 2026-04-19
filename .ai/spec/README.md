# OpenShift LightSpeed Service -- Specifications

These specs define the requirements, behaviors, and architecture for the OLS lightspeed-service. They are organized into two layers:

- **[`what/`](what/README.md)** -- Behavioral rules: WHAT the system must do and WHY. Technology-neutral, testable assertions. Use these to understand requirements, fix bugs, or rebuild components.
- **[`how/`](how/README.md)** -- Architecture specs: HOW the current implementation is structured. Module boundaries, data flow, design patterns. Use these to navigate, modify, and extend the codebase.

## Scope

These specs cover the **lightspeed-service** Python application only. The operator, console plugin, and RAG content pipeline are separate projects. Jira data covering all projects is in `.ai/jira-*.md` files.

## Audience

AI agents (Claude). Specs optimize for precision, unambiguous rules, and machine-parseable structure.

## Quick Start

| I want to... | Read |
|--------------|------|
| Understand what OLS does | `what/system-overview.md` |
| Fix a bug in query processing | `what/query-processing.md` + `how/query-pipeline.md` |
| Add a new LLM provider | `what/llm-providers.md` + `how/llm-providers.md` |
| Understand the API | `what/api.md` |
| Navigate the codebase | `how/project-structure.md` |
| See what's planned | Look for `[PLANNED: OLS-XXXX]` in `what/` specs |

## Conventions

- `[PLANNED: OLS-XXXX]` markers in `what/` specs indicate existing rules about to change due to open Jira work
- "Planned Changes" sections list new capabilities not yet in code
- Config field names reference `olsconfig.yaml` paths (e.g., `ols_config.tool_filtering.threshold`)
- Internal constants are stated as behavioral rules without numeric values; `how/` specs may include specific values

## Project History

OpenShift Lightspeed evolved through these phases:

1. **Prototype** (Q4 2023): Basic chat with OpenAI + RAG from product docs
2. **Early Access** (Q1-Q2 2024): Configuration system, conversation cache, auth, multi-provider, metrics, feedback
3. **Tech Preview** (Q3 2024): Streaming, RHOAI/RHELAI providers, Azure Entra ID, TLS profiles
4. **GA** (Q4 2024 - Q1 2025): PostgreSQL cache, quota management, FIPS, disconnected mode, transcripts
5. **Post-GA** (2025-2026): MCP tools, context-aware queries, skills, tool approval, BYOK, custom system prompts, query modes, dynamic config reload

Jira tracking: Features in OCPSTRAT project (label: OLS), Epics/Stories in OLS project.
