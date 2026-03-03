---
name: raise-pr
description: Step-by-step workflow for committing staged changes and opening a pull request in the lightspeed-service repo.
disable-model-invocation: true
---

# Create a PR

## Step 1: Pre-commit checks

Run in order and fix all failures before proceeding:

1. `make test-unit`
2. `make test-integration`
3. `make verify`
4. Confirm 90%+ coverage is maintained for changed code

## Step 2: Gather context

If a Jira issue ID has not been provided, ask the user for it before proceeding.

Run these in parallel to understand the changes:

```bash
git status
git diff HEAD
git log --oneline -10
```

## Step 3: Commit

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

## Step 4: Push

```bash
git push -u origin HEAD
```

## Step 5: Open the PR

Title format: `<JIRA-ID>: <same short imperative sentence as commit>`

```
OLS-2681: Remove question validation subsystem
```

Read `.github/PULL_REQUEST_TEMPLATE.md` and populate all sections:

- **Description** — what the change does and why, derived from the diff
- **Type of change** — tick the relevant checkbox(es)
- **Related Tickets & Documents** — `Closes #<github-issue>` if it resolves one; always link the Jira ticket
- **Checklist** — tick what applies; leave unticked items that genuinely don't apply
- **Testing** — describe how the change was verified

```bash
gh pr create --title "<JIRA-ID>: <description>" --body "$(cat <<'EOF'
<populated template>
EOF
)"
```

Return the PR URL to the user when done.
