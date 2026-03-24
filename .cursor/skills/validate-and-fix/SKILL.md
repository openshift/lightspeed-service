---
name: validate-and-fix
description: Run the full validation pipeline (make test-unit, make verify, make test-integration) and auto-fix trivial failures like black formatting, unused imports, and ruff-fixable lint errors. Use when the user asks to validate, run tests, check the pipeline, or verify changes are clean.
disable-model-invocation: true
---

# Validate & Auto-Fix

Run the project validation pipeline, auto-fix trivial issues, and re-run until green or a real failure is found.

## Rules

- Never modify production logic to fix a test. Only fix test expectations, imports, formatting, and lint.
- Never skip or delete a failing test.
- Stop after 3 auto-fix cycles to avoid loops.
- Report real failures clearly; do not attempt speculative fixes.

## Step 0: Ensure Dependencies Are Installed

```bash
make install-deps 2>&1 | tail -10
```

This ensures the virtualenv matches the lockfile. Skipping this step is the most common cause of spurious import errors (e.g. `llama-index-embeddings-huggingface` not found despite being in requirements.txt).

## Step 1: Run Unit Tests

```bash
make test-unit 2>&1 | tail -40
```

If all pass, proceed to Step 3 (integration tests).
If failures occur, classify each failure (see Step 2).

## Step 2: Classify and Fix Failures

For each failure, determine its type:

**Auto-fixable** (fix immediately, then re-run Step 1):

| Type | Fix |
|------|-----|
| Missing `await` on async call | Add `await`, mark test `async def`, add `@pytest.mark.asyncio` |
| Test asserts old default value | Update assertion to match new default |

**Real failures** (do not auto-fix):

- Logic errors in production code
- Assertion failures reflecting actual behavior changes
- Import errors from missing dependencies
- Failures in code you did not modify

For real failures: report the test name, file, and error summary, then stop.

## Step 3: Run Integration Tests

```bash
make test-integration 2>&1 | tail -20
```

Integration test failures are almost always real failures. Report and stop.

## Step 4: Run Verify

```bash
make verify 2>&1 | tail -30
```

This runs: black --check, ruff, woke, pylint, mypy.

If failures occur, apply the same classify-and-fix logic from Step 2.
Common fixes at this stage:

| Type | Fix |
|------|-----|
| Black formatting | `uv run black .` |
| Unused import (ruff F401) | Remove the import |
| Unused variable (ruff F841) | Remove the assignment |
| Unused noqa directive (RUF100) | Remove the noqa comment |
| Other ruff-fixable | `uv run ruff check . --fix` |
| Pylint issues from changed code | Fix inline |

Re-run `make verify` after each fix. Proceed to Step 5 when green.

## Step 5: Report

Report exactly:

- `make test-unit`: X passed / Y failed
- `make verify`: pass/fail (list any remaining issues)
- `make test-integration`: X passed / Y failed
- Auto-fixes applied (list each: file, what was fixed)
- Cycles used: N/3

Do not include unrelated diagnostics or suggestions.
