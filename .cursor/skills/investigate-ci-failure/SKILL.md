---
name: investigate-ci-failure
description: Investigate CI/Prow job failures on a GitHub pull request. Use when the user pastes a PR URL and asks about CI failures, red checks, test failures, or wants to understand why a job failed.
disable-model-invocation: true
---

# Investigate CI Failure

Given a PR URL (e.g. `https://github.com/openshift/lightspeed-service/pull/2825`), diagnose why CI jobs failed.

## Workflow

### 1. Extract PR info

Parse org, repo, and PR number from the URL. Fetch metadata with `gh`:

```bash
# PR metadata
gh api repos/{org}/{repo}/pulls/{pr} --jq '{title, state, user: .user.login, head_sha: .head.sha}'

# Changed files
gh api repos/{org}/{repo}/pulls/{pr}/files --jq '.[].filename'
```

### 2. Get check statuses

```bash
# All checks at a glance
gh pr checks {pr} --repo {org}/{repo}

# Detailed statuses with Prow URLs (use head SHA from step 1)
gh api repos/{org}/{repo}/statuses/{head_sha} \
  --jq '.[] | select(.state == "failure" or .state == "error") | {context, state, target_url}'
```

This gives you the list of failed jobs and their Prow dashboard URLs.

### 3. Construct GCS artifact URLs

From a Prow `target_url` like:
```
https://prow.ci.openshift.org/view/gs/test-platform-results/pr-logs/pull/{org}_{repo}/{pr}/{job_name}/{build_id}
```

Derive:
- **Directory browser** (for navigating artifact tree):
  `https://gcsweb-ci.apps.ci.l2s4.p1.openshiftapps.com/gcs/test-platform-results/pr-logs/pull/{org}_{repo}/{pr}/{job_name}/{build_id}/`
- **Raw file content** (for fetching logs and JSON):
  `https://storage.googleapis.com/test-platform-results/pr-logs/pull/{org}_{repo}/{pr}/{job_name}/{build_id}/{path}`

### 4. Triage the failure

For each failed job, fetch artifacts in this order:

#### 4a. Quick status

```
GET storage.googleapis.com/.../finished.json
```

Check `"passed": false` and `"result": "FAILURE"`.

#### 4b. Build log (most useful)

```
GET storage.googleapis.com/.../build-log.txt
```

This is the main ci-operator build log. It can be large (200KB+). Search from the **end** for:
- `failed` / `FAILED` / `error` / `ERROR`
- `step .* failed`
- Python tracebacks (`Traceback`, `AssertionError`, `FAILED tests/`)
- Container crash indicators (`CrashLoopBackOff`, `OOMKilled`, `Error from server`)

#### 4c. Artifact tree

Browse the artifact directory to find step-specific logs:

```
GET gcsweb-ci.apps.ci.l2s4.p1.openshiftapps.com/gcs/.../artifacts/
```

Typical structure:
```
artifacts/
├── {job-step-name}/
│   └── {step}/
│       ├── build-log.txt     ← step-specific log
│       ├── finished.json
│       └── artifacts/        ← test results, JUnit XML, etc.
├── ci-operator.log
├── junit_operator.xml
└── ci-operator-step-graph.json
```

For e2e failures, drill into `artifacts/{step-name}/e2e/build-log.txt` for the actual test output.

When multiple jobs have failed, investigate each in a separate subagent (Task tool)
to keep build-log context isolated and run fetches in parallel.

#### 4d. JUnit results (if available)

Look for `junit*.xml` files in artifact directories. These list individual test cases with pass/fail and failure messages. Fetch them via `storage.googleapis.com` URLs.

### 5. Cross-reference with PR changes

Compare the failure with the files changed in the PR. Common patterns:

| Failure type | Likely cause |
|---|---|
| Unit/integration test failure | Direct code bug in changed files |
| e2e cluster test failure | Infrastructure issue OR deployment-breaking change |
| Verify/lint failure | Formatting, type errors, or import issues |
| Image build failure | Dependency or Dockerfile issue |
| Flaky (passes on retest) | Known flake, not PR-related |

Check if the same job fails on `main` branch (flaky test) by looking at job history:
```
https://prow.ci.openshift.org/job-history/gs/test-platform-results/pr-logs/directory/{job_name}
```

### 6. Report findings

Summarize:
1. **Which jobs failed** and which passed
2. **Root cause** for each failure (with relevant log excerpts)
3. **Whether it's PR-related or infrastructure/flaky**
4. **Suggested fix** if the failure is caused by the PR changes

## Known CI jobs for this repo

| Context | What it tests |
|---|---|
| `ci/prow/unit` | `make test-unit` — pytest unit tests |
| `ci/prow/integration` | `make test-integration` — integration tests |
| `ci/prow/verify` | `make verify` — black, ruff, pylint, mypy, woke |
| `ci/prow/security` | `make security-check` — bandit |
| `ci/prow/images` | Container image build |
| `ci/prow/fips-image-scan-service` | FIPS compliance scan |
| `ci/prow/e2e-ols-cluster` | Full cluster e2e — deploys OLS + operator on OpenShift, runs `make test-e2e` |
| `tide` | Merge readiness (labels, approvals) — not a test |
| Konflux | Supply chain security pipeline (separate from Prow) |

## Tool usage notes

- Use `gh` CLI for all GitHub API calls (PR metadata, statuses, checks, comments, files).
- Use `WebFetch` to browse GCS directories (`gcsweb-ci.apps.ci.l2s4.p1.openshiftapps.com/gcs/...`).
- Use `WebFetch` to fetch raw log/JSON content (`storage.googleapis.com/test-platform-results/...`).
- The Prow dashboard URL itself is JS-rendered and not useful via WebFetch — always use GCS URLs instead.
- Build logs can be very large. When fetched via WebFetch, they're saved to a temp file — read from the end to find failures quickly.
