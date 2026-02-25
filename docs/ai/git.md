# Git Workflow Guide

Read this when creating commits or pull requests.

## Commit Messages

Follow the style of recent commits in this repo — short imperative sentence, no period:

```
Fix spurious GeneratorExit errors in LangChain streaming logs
Add quota enforcement for WatsonX provider
Bump memory limits for SAST Konflux tasks
```

- No conventional commit prefixes (`feat:`, `fix:`) — plain imperative sentences
- Keep to a single line unless the change genuinely needs more explanation
- Reference the issue number in the PR, not in the commit message

## Creating a Pull Request

The PR template is at `.github/PULL_REQUEST_TEMPLATE.md`. Populate all sections:

- **Description** — what the change does and why
- **Type of change** — tick the relevant checkbox(es)
- **Related Tickets & Documents** — always link the related issue (`Closes #<number>` if it resolves it)
- **Checklist** — tick what applies; leave unticked items that genuinely don't apply
- **Testing** — describe how the change was verified; include relevant command output or screenshots

## Before Committing

Run in order and fix all failures before committing:

1. `make verify` — linters and type checks
2. `make test-unit` — unit tests
3. Confirm 90%+ coverage is maintained for changed code
