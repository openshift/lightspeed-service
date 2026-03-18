---
name: find-duplication
description: Find code duplication in the codebase. Supports two modes - scoped to current branch changes or a full codebase sweep. Use when the user asks to find duplicated code, copy-paste, repeated patterns, or wants to deduplicate before a PR.
disable-model-invocation: true
---

# Find Code Duplication

Detect duplicated or near-duplicate code and suggest consolidation candidates.

## Rules

- Report findings, do not refactor. Refactoring is a separate task.
- Focus on production code (`ols/`). Skip test duplication unless explicitly asked.
- Group findings by severity: exact duplicates first, then near-duplicates.
- For each finding, state whether extraction is worth it or acceptable duplication.

## Step 1: Determine Scope

Ask the user:
- **Branch mode**: only files changed in the current branch vs main.
- **Full mode**: scan the entire `ols/` directory.

For branch mode:

```bash
git diff --name-only origin/main -- 'ols/' | grep '\.py$'
```

For full mode, the target is `ols/`.

## Step 2: Run Pylint Duplicate Detection

```bash
uv run pylint --disable=all --enable=duplicate-code --min-similarity-lines=6 <target files or directory>
```

Review output. Filter out false positives:
- Import blocks (common imports are not duplication)
- Pydantic model boilerplate (Field declarations)
- Single-line patterns (logging, raises)

## Step 3: Semantic Duplication Search

Pylint only catches textual similarity. Also look for:

1. **Similar function signatures** — functions with near-identical parameter lists doing similar work.
2. **Repeated error handling** — same try/except/log/return pattern across multiple files.
3. **Copy-pasted blocks** — search for distinctive string literals or variable names that appear in multiple files.

```bash
rg "<distinctive pattern>" ols/ --type py -l
```

## Step 4: Classify Findings

For each duplicate found, classify:

| Category | Action |
|----------|--------|
| **Extract** — identical logic in 3+ places | Recommend a shared helper |
| **Parameterize** — same structure, different values | Recommend a common function with parameters |
| **Acceptable** — similar but serving different domains | Note it, no action needed |
| **Test-only** — repeated test setup/fixtures | Recommend shared fixture (only if user asked) |

## Step 5: Report

For each finding:

1. Files and line ranges involved
2. What is duplicated (brief description)
3. Classification (extract / parameterize / acceptable)
4. Suggested location for shared code (if applicable)

Summary: total findings, how many actionable, estimated lines saved.
