---
name: review-pr
description: Review PR with structured approach covering architecture, naming, patterns, and critical questions
disable-model-invocation: true
---

# Review PR

When asked to review a PR, follow this structured approach.

## 1. Fetch Latest Changes

**Always** fetch the latest PR state before reviewing. The cached PR data may be stale.

```bash
git fetch upstream pull/<PR_NUMBER>/head:pr-<PR_NUMBER>
git log pr-<PR_NUMBER> --oneline -10
git diff upstream/main...pr-<PR_NUMBER> --stat
```

For follow-up reviews, re-fetch to get new commits:

```bash
git fetch upstream pull/<PR_NUMBER>/head:pr-<PR_NUMBER> --force
```

Read diffs per area (endpoints, models, utils, tests) rather than one massive diff.

## 2. Understand What It Implements

- Summarize the feature/fix in 2-3 sentences
- Identify the flow: entry point -> processing -> output
- Map which files serve which role (endpoint, model, utility, config)

## 3. Evaluate How It's Implemented

- **Architecture**: Is logic in the right layer? (endpoints vs utils vs models)
- **Data flow**: How does data travel through the system? Any unnecessary transformations?
- **Error handling**: Are errors caught at the right level? Silent failures vs exceptions?
- **Duplication**: Is code copy-pasted across files?
- Is there a dead code
- Relevant documentation updates

## 4. Assess Naming

- Do file names match what they contain?
- Do function/class names describe their behavior?
- Are types self-documenting or do they need aliases?
- Is naming consistent with existing codebase patterns?

## 5. Check Pythonic Patterns

- Pydantic models for validation instead of hand-rolled parsing
- Type hints on all public functions (avoid `Any` where possible)
- `TypeAlias` for complex repeated types
- Proper use of framework features (FastAPI dependencies, Pydantic validators)
- No JSON-inside-JSON when native types work
- Imports at the top (justification is required otherwise)

## 6. Ask Critical Questions

Focus on design decisions that have long-term impact:

- Why was this approach chosen over alternatives?
- What happens when input is invalid/missing/malformed?
- Are there security implications? (token logging, size limits, injection)
- Do tests verify behavior or just status codes?
- Is the API contract clear to consumers? (OpenAPI schema, examples)
- Is PR bundling other/unrelated changes to its main goal, that should be separated as separate PRs - or at least commits?

## 7. Output Format

Structure the review as:

1. **Summary**: What the PR does (2-3 sentences)
2. **File-by-file analysis**: Role of each changed file
3. **Issues table**: Priority (must-fix / should-fix / nice-to-have), issue, location
4. **What's good**: Acknowledge well-done aspects
5. **Critical questions**: Design decisions that need answers
6. **Confidence notes**: Flag each concern with a confidence level (certain / likely / unsure).
7. **Closing reminder**: Always end the review with: *"You, as a human, need to evaluate these AI findings instead of just copying them as review comments. That would just shift the responsibility of validation to the PR creator."*
