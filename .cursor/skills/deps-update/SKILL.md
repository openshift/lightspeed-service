---
name: deps-update
description: >-
  Update Python dependencies to latest versions using uv,
  regenerate lock and requirements.txt, then verify linting
  and tests pass. Fix breakage from API changes in bumped
  packages. Use when the user says "deps update",
  "bump dependencies", or "update deps".
---

# deps-update

## Step 1: Snapshot Current State

Before changing anything:

1. Fetch upstream and check out a clean base:
   ```bash
   git fetch upstream
   git checkout upstream/main
   git checkout -b chore/deps-update
   ```
   If the working tree has uncommitted changes, **stop and
   tell the user** to commit or stash first. Dependency
   updates must land on a clean tree.
2. Run `uv lock --check` to confirm the lock file is
   currently consistent
3. Run `uv run python --version` to record the Python version

## Step 2: Bump Dependencies

If a specific package was requested, bump only that package:

```bash
uv lock --upgrade-package {package}
uv sync --group dev --extra evaluation
```

Otherwise bump everything:

```bash
uv lock --upgrade
uv sync --group dev --extra evaluation
```

Capture the output of `uv lock --upgrade` — it lists all
version changes and is needed for the commit message and
report.

After running `uv lock --upgrade`, check the output for
warnings. If a warning indicates a missing extra or
incompatible constraint, investigate whether it affects
this project. Note any benign warnings in the final report.

Then regenerate the pinned requirements file:

```bash
make requirements.txt
```

This runs `uv export` under the hood and produces a
`requirements.txt` with hashes for production packages.

## Step 3: Verify — Lint and Type Checks

Run the full verification suite:

```bash
make format
make verify
```

If verification passes cleanly, proceed to Step 4.

If verification fails:

1. Read the error output carefully
2. Identify which bumped package caused the breakage
   (often a renamed/removed API, changed type signatures,
   or new deprecation-as-error)
3. Fix the calling code to match the new API — do not
   pin the package back to the old version unless the new
   version has a confirmed regression
4. Re-run `make format && make verify`
5. Repeat until clean

## Step 4: Verify — Unit Tests

Run the unit test suite:

```bash
make test-unit
```

If tests pass, proceed to Step 5.

If tests fail:

1. Distinguish between **test breakage from API changes**
   (assertions on old behavior, mocks of renamed methods)
   and **genuine regressions** (new bugs introduced by
   a library update)
2. For API changes — update tests and source code to
   match the new API
3. For genuine regressions — check the library's changelog
   and issue tracker. If confirmed upstream bug, pin that
   specific package to the last working version in
   `pyproject.toml` and re-run `uv lock && uv sync`
4. Re-run `make test-unit`
5. Repeat until clean

## Step 5: Verify — Integration Tests

Run the integration test suite:

```bash
make test-integration
```

Same triage approach as Step 4 if failures occur.

## Step 6: Report, Commit, and PR

Check which files changed beyond the dependency files:

```bash
git diff --name-only
```

Always present a summary to the user before committing:

- Number of packages bumped, with notable version jumps
  (major versions, security-relevant packages)
- Verification status (lint, unit tests, integration tests)
- List of all changed files
- Any warnings from Step 2 that were deemed benign
- Any source/test files modified to fix API changes
  (with a brief explanation of each fix)

**If only `pyproject.toml`, `uv.lock`, and `requirements.txt`
changed** — commit and raise a PR automatically without
asking:

```bash
git add pyproject.toml uv.lock requirements.txt
git commit -m "chore: bump dependencies to latest"
```

Then follow the `raise-pr` skill to open the PR.

**If source or test files were also modified** (API change
fixes from Steps 3–5) — wait for user acknowledgment
before committing. Then commit all changes and follow the
`raise-pr` skill.

## Constraints

- **Clean tree required** — do not start if there are
  uncommitted changes.
- **Fix forward, not back** — prefer updating code to
  match new APIs over pinning old versions. Only pin back
  when there's a confirmed upstream regression.
- **Do not skip verification** — all four gates (lint,
  types, unit tests, integration tests) must pass before
  committing.
- **Do not touch unrelated code** — only modify files
  that break due to the dependency bump. No drive-by
  refactors.
- **Report honestly** — if a test failure looks unrelated
  to the bump, say so. Let the user decide whether to
  investigate separately.
- **No manual requirements.txt edits** — always regenerate
  via `make requirements.txt`, never hand-edit.
