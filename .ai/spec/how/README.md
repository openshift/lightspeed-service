# Architecture Specifications (how/)

These specs describe HOW the OLS service is structured -- module boundaries, data flow, design patterns, key abstractions, and implementation decisions. They are grounded in the current Python/FastAPI codebase and should be updated when the code changes.

## Spec Index

| Spec | Description |
|------|-------------|
| [project-structure.md](project-structure.md) | Directory layout, module responsibilities, startup sequence, layer integration |
| [query-pipeline.md](query-pipeline.md) | DocsSummarizer orchestration, stage-by-stage data flow, token budget algorithm, streaming interleaving |
| [llm-providers.md](llm-providers.md) | Provider registry, decorator discovery, class hierarchy, parameter mapping, HTTP client setup |
| [tools.md](tools.md) | MCP client lifecycle, tool gathering pipeline, approval state machine, retry/backoff, parallel execution |
| [config.md](config.md) | AppConfig singleton, Pydantic class hierarchy, two-phase validation, dynamic reload mechanism |
| [cache.md](cache.md) | Abstract cache interface, PostgreSQL schema, advisory locks, JSON serialization, in-memory LRU |

## When to Read These

- **Navigating the codebase**: Start with `project-structure.md` to understand where things live.
- **Modifying a subsystem**: Read the relevant `how/` spec to understand the current architecture before making changes.
- **Adding a new provider/cache backend/MCP server**: The `how/` specs include step-by-step guides for extension points.
- **Debugging**: The data flow sections trace the exact path requests take through the code.

## Relationship to what/ Specs

The [`what/` specs](../what/README.md) define behavioral contracts (technology-neutral). These `how/` specs describe the implementation that fulfills those contracts. When the two diverge, the `what/` spec is the source of truth for correct behavior, and the `how/` spec should be updated to reflect the current code.
