---
name: raise-pr
description: Step-by-step workflow for committing staged changes and opening a pull request in the lightspeed-service repo.
disable-model-invocation: true
---

# Create a PR

## Step 1: Branch

```bash
git fetch upstream
git branch --show-current
git log --oneline upstream/main..HEAD
git diff upstream/main --stat
gh pr list --head "$(git branch --show-current)" --state open
```

Present the summary and ask before proceeding:

> "You are currently on branch `<branch>`. It has `<N>` commit(s) ahead of `upstream/main` touching: `<files>`.
> [If an open PR exists]: This branch already has an open PR: `<url>`. Pushing here would add commits to that existing PR.
> Should I create a **new branch** from `upstream/main` (recommended), or use the current branch?"

**Do not continue until the user answers.**

If creating a new branch — git carries uncommitted changes automatically:

```bash
git fetch upstream
git checkout -b <type>/<short-description> upstream/main
# cherry-pick intended commits from the old branch if needed
git cherry-pick <sha1> <sha2> ...
```

Never branch from `origin/main` — the fork may have stale commits that will appear in the PR.

## Step 2: Pre-commit checks

Run in order and fix all failures before proceeding:

1. `make test-unit`
2. `make test-integration`
3. `make verify`
4. Confirm 90%+ coverage is maintained for changed code

## Step 3: Commit and push

If a Jira issue ID has not been provided, ask the user for it before proceeding.

Commit message: short imperative sentence, no period, no conventional prefix, no Jira ID.

```bash
git add -A   # or only the relevant files
git commit -m "<message>"
git push -u origin HEAD
```

## Step 4: Open the PR

Title format: `<JIRA-ID>: <same short imperative sentence as commit>`

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
