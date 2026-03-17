---
name: rebase-clean
description: Performs a strict clean rebase of a feature branch onto main with minimal conflict resolution and full validation. Use when the user asks to rebase carefully, avoid extra branches, avoid exploratory edits, and run make test-unit, make test-integration, and make verify until green.
disable-model-invocation: true
---

# Clean Rebase Workflow

Use this workflow exactly for rebases in this repository.

## Rules

- Do not create extra temporary branches unless the user explicitly asks.
- Do not make unrelated edits.
- Do not stop after analysis; finish rebase and validation.
- Keep output brief and operational.
- Never assume target branch names; detect current branch first.

## Step 0: Detect Branch Context

1. Run `git branch --show-current`.
2. Treat detected current branch as the candidate target branch.
3. Ask user to confirm explicitly before proceeding:
   - `I detected '<current-branch>' as the current branch. Should I rebase this branch?`
4. If user specified a target branch, verify it matches current branch; if not, stop and ask.
5. Do not run reset/rebase commands until this confirmation is received.
6. Before any reset/rebase command, restate:
   - current branch
   - target branch
   - approved baseline ref
   and confirm when there is any mismatch or ambiguity.

## Step 1: Restore Branch Baseline (Rerun Only)

Use this step only when a prior rebase attempt failed or introduced unwanted edits.
For a first rebase attempt, skip this step and go directly to Step 2.

1. Checkout target branch.
2. Hard reset to the known backup/base commit the user approved.
3. Fetch `origin`.

Command pattern:

```bash
git checkout <branch>
git reset --hard <backup-or-approved-base>
git fetch origin
```

## Step 2: Rebase Onto Main

Run:

```bash
git rebase origin/main
```

If conflicts occur, resolve only the conflicted files. Do not add extra refactors.

## Step 3: Conflict Resolution Policy

For each conflicted file:

1. Start from one side (`ours` or `theirs`) as a temporary base.
2. Apply only minimal compatibility changes required to compile and run tests on the new base branch.
3. Keep behavior from the feature commit unless it is incompatible with upstream removals/renames.
4. Resolve matching tests together with production code changes.
5. Avoid opportunistic refactors during conflict resolution.

Then:

```bash
git add <resolved files>
GIT_EDITOR=true git rebase --continue
```

Repeat until rebase completes.

## Step 4: Verify No Code Loss

Run these checks against the approved pre-rebase baseline (backup/base commit):

```bash
git range-diff origin/main...<approved-base> origin/main...HEAD
git diff --name-status <approved-base>..HEAD
git merge-base --is-ancestor origin/main HEAD
git log --left-right --cherry-pick --oneline origin/main...HEAD
```

Acceptance criteria:

1. Branch intent is preserved:
   - Commit intent maps cleanly in `range-diff` (rewritten SHAs are fine).
2. Main is fully incorporated:
   - `git merge-base --is-ancestor origin/main HEAD` succeeds (exit code 0).
3. No unexpected file removals/renames beyond what upstream already changed.
4. Any conflict-touched files have expected compatibility-only deltas.
5. No unexplained drift from the approved baseline.

If any check fails, treat as code-loss risk and restart from Step 1.

## Step 5: Full Validation Pipeline

Run exactly:

```bash
make test-unit && make test-integration && make verify
```

If validation or code-loss checks fail at any point:

1. Treat it as an incorrect rebase result.
2. Abort/restore to the approved baseline.
3. Re-run the rebase/conflict-resolution flow from Step 1.
4. Re-run Step 4 and this step.
5. Repeat full loop until all checks pass.

## Step 6: Final Report

Report only:

- Rebase completed (yes/no)
- Current branch and ahead/behind state
- `make test-unit` result
- `make test-integration` result
- `make verify` result

Do not include unrelated diagnostics.
