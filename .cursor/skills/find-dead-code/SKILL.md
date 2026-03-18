---
name: find-dead-code
description: Find unused functions, classes, imports, and unreachable code paths. Use when the user asks to find dead code, unused code, cleanup candidates, or wants to reduce codebase size.
disable-model-invocation: true
---

# Find Dead Code

Detect unused code that can be safely removed.

## Rules

- Report findings, do not delete. Removal is a separate task.
- Focus on production code (`ols/`). Skip tests unless explicitly asked.
- Vulture has false positives — classify each finding before recommending removal.
- Code used only via dynamic dispatch (e.g. Pydantic validators, FastAPI dependencies) is not dead.

## Step 1: Determine Scope

Ask the user:
- **Branch mode**: only files changed in the current branch vs main.
- **Full mode**: scan the entire `ols/` directory.

For branch mode:

```bash
git diff --name-only origin/main -- 'ols/' | grep '\.py$'
```

## Step 2: Run Vulture

```bash
uvx vulture <target> --min-confidence 80
```

`--min-confidence 80` reduces noise. Lower confidence findings are more likely false positives.

## Step 3: Run Pylint Unused Checks

```bash
uv run pylint --disable=all --enable=unused-import,unused-variable,unused-argument,unreachable <target>
```

Cross-reference with vulture findings to increase confidence.

## Step 4: Filter False Positives

Common false positives in this codebase:

| Pattern | Why it's not dead |
|---------|-------------------|
| Pydantic `model_validator`, `field_validator` | Called by Pydantic, not directly |
| FastAPI dependency functions | Injected via `Depends()` |
| `__eq__`, `__hash__`, `__str__` | Called implicitly by Python |
| Constants used in config YAML | Referenced by config loader |
| Abstract method implementations | Called via base class interface |
| Imports re-exported from `__init__.py` | Used by external consumers |

## Step 5: Classify Findings

For each finding, classify:

| Category | Criteria | Action |
|----------|----------|--------|
| **Remove** | Clearly unused, no dynamic references | Safe to delete |
| **Verify** | Possibly used dynamically or externally | Search for string references before removing |
| **False positive** | Pydantic/FastAPI/magic method | Skip |

For "Verify" findings, search for string-based references:

```bash
rg "<function_or_class_name>" ols/ tests/
```

## Step 6: Report

For each finding:

1. File and line number
2. What is unused (function, class, import, variable)
3. Confidence level (vulture %)
4. Classification (remove / verify / false positive)

Summary: total findings, how many safe to remove, estimated lines saved.
