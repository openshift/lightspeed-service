---
name: safe-code-change
description: After a code change, find affected tests, update them to match new behavior, then run the full validation pipeline once. Use when the user has made or asked for a code change and wants to make sure nothing is broken.
disable-model-invocation: true
---

# Safe Code Change

After a code change is made, find and fix affected tests before running validation.

## Rules

- The code change is already done. Do not modify production code.
- Only update tests to match the new behavior, not the other way around.
- Do not reformat or lint-fix during test updates. Save that for validation.
- If a test change is ambiguous (unclear what the new expected behavior is), ask the user.

## Step 1: Identify What Changed

```bash
git diff --name-only
git diff --stat
```

List the modified production files (ignore test files, configs, docs).

## Step 2: Find Affected Tests

Search for imports of changed functions/classes across all test files:

```bash
rg "from ols\.<changed_module> import" tests/
```

## Step 3: Analyze Impact on Tests

For each affected test file, check whether the change breaks existing tests:

1. **Signature changes** — function renamed, parameters added/removed/reordered.
2. **Behavior changes** — return value, side effects, exceptions differ.
3. **Removed code** — tests for deleted functions/classes need removal.
4. **New code** — consider whether new tests are needed (ask user if unclear).

## Step 4: Update Tests

Apply minimal fixes to each affected test:

- Update function names, parameters, expected values.
- Add `async`/`await` and `@pytest.mark.asyncio` if sync changed to async.
- Remove tests for deleted functionality.
- Do not add new tests unless the user asks.

## Step 5: Done

Report what was updated. The user can invoke validate-and-fix when ready.
