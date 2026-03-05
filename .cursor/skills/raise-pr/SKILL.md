---
name: raise-pr
description: Step-by-step workflow for committing staged changes and opening a pull request in the lightspeed-service repo.
disable-model-invocation: true
---

# Create a PR

## Step 1: Verify branching

Run:

```bash
git fetch upstream
git branch --show-current
git log --oneline upstream/main..HEAD
git diff upstream/main --stat
gh pr list --head "$(git branch --show-current)" --state open
```

**The default is always a new branch from `upstream/main`.** Only reuse the current branch if the user explicitly confirms it. Never branch from `origin/main` — the fork may have stale commits that will appear in the PR.

Present this summary and ask before proceeding:

> "You are currently on branch `<branch>`. It has `<N>` commit(s) ahead of `upstream/main` touching: `<files>`.
> [If an open PR exists]: ⚠️ This branch already has an open PR: `<url>`. Pushing here would add commits to that existing PR.
> Should I create a **new branch** from `upstream/main` (recommended), or use the current branch?"

**Do not continue until the user answers.**

If creating a new branch:

```bash
git fetch upstream
git checkout -b <short-description> upstream/main
# cherry-pick intended commits from the old branch if needed
git cherry-pick <sha1> <sha2> ...
```

Also sync the fork's `main` to keep it clean:

```bash
git checkout main
git reset --hard upstream/main
git push --force-with-lease origin main
git checkout <new-branch>
```

Branch naming: `<type>/<short-description>` — e.g. `feat/add-mcp-stdio-server`, `fix/token-quota-reset`.

## Step 2: Pre-commit checks

Run in order and fix all failures before proceeding:

1. `make test-unit`
2. `make test-integration`
3. `make verify`
4. Confirm 90%+ coverage is maintained for changed code

## Step 3: Gather context

If a Jira issue ID has not been provided, ask the user for it before proceeding.

Run these in parallel to understand the changes:

```bash
git status
git diff HEAD
git log --oneline -10
```

## Step 4: Commit

Write a short imperative sentence, no period, no conventional prefix:

```
Remove question validation subsystem
Fix spurious GeneratorExit errors in LangChain streaming logs
Add quota enforcement for WatsonX provider
```

- Single line unless the change genuinely needs more explanation
- Do not reference the Jira/issue number in the commit message

```bash
git add -A   # or only the relevant files
git commit -m "<message>"
```

## Step 5: Push

```bash
git push -u origin HEAD
```

## Step 6: Open the PR

Title format: `<JIRA-ID>: <same short imperative sentence as commit>`

```
OLS-2681: Remove question validation subsystem
```

Read `.github/PULL_REQUEST_TEMPLATE.md` and populate all sections:

- **Description** — what the change does and why, derived from the diff
- **Type of change** — tick the relevant checkbox(es)
- **Related Tickets & Documents** — `Closes #<github-issue>` if it resolves one; always link the Jira ticket
    - Ticket links are in the format: https://issues.redhat.com/browse/<JIRA-ID>
- **Checklist** — tick what applies; leave unticked items that genuinely don't apply
- **Testing** — describe how the change was verified

```bash
gh pr create --title "<JIRA-ID>: <description>" --body "$(cat <<'EOF'
<populated template>
EOF
)"
```

Return the PR URL to the user when done.
