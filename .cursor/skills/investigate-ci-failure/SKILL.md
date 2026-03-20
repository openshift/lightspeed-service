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

#### 4c. Artifact tree exploration

The build log alone often doesn't tell the full story. Browse the GCS artifact directory
to find step-specific logs, cluster state, and pod logs:

```
GET gcsweb-ci.apps.ci.l2s4.p1.openshiftapps.com/gcs/.../artifacts/
```

Full artifact tree for an e2e job:
```
{build_id}/
├── build-log.txt                    ← main ci-operator log (start here)
├── finished.json                    ← pass/fail + metadata
├── artifacts/
│   ├── ci-operator.log              ← detailed ci-operator log
│   ├── junit_operator.xml           ← top-level JUnit results
│   ├── ci-operator-step-graph.json  ← step dependency graph
│   ├── ci-operator-metrics.json
│   ├── metadata.json
│   ├── build-logs/                  ← container image build logs
│   │   ├── lightspeed-service-api-amd64.log
│   │   ├── root-amd64.log
│   │   └── src-amd64.log
│   ├── build-resources/             ← CI namespace state
│   │   ├── pods.json                ← all pods in CI namespace
│   │   ├── events.json              ← k8s events (useful for crashes)
│   │   ├── builds.json
│   │   ├── imagestreams.json
│   │   └── clusterClaim.json
│   ├── release/                     ← cluster provisioning step
│   │   ├── build-log.txt
│   │   └── finished.json
│   └── e2e-ols-cluster/             ← test workflow steps
│       ├── ipi-install-rbac/        ← cluster RBAC setup
│       │   └── build-log.txt
│       ├── e2e/                     ← THE ACTUAL TEST STEP
│       │   ├── build-log.txt        ← test runner output (pytest)
│       │   ├── finished.json
│       │   └── artifacts/           ← per-provider test results
│       │       ├── junit_e2e_azure_openai.xml
│       │       ├── junit_e2e_openai.xml
│       │       ├── junit_e2e_watsonx.xml
│       │       ├── junit_e2e_rhelai_vllm.xml
│       │       ├── junit_e2e_rhoai_vllm.xml
│       │       ├── junit_e2e_*_tool_calling.xml
│       │       ├── junit_e2e_quota_limits.xml
│       │       └── {provider}/cluster/   ← cluster state per provider
│       │           ├── podlogs/
│       │           │   ├── lightspeed-app-server-*.log  ← OLS service logs
│       │           │   ├── lightspeed-postgres-server-*.log
│       │           │   └── lightspeed-console-plugin-*.log
│       │           ├── olsconfig.yaml    ← OLS config used
│       │           ├── pods.yaml
│       │           ├── deployments.yaml
│       │           ├── configmap.yaml
│       │           ├── services.yaml
│       │           └── routes.yaml
│       ├── gather-must-gather/      ← cluster diagnostics
│       │   └── artifacts/
│       │       ├── must-gather.tar  ← full must-gather (large, ~25MB)
│       │       ├── camgi.html       ← must-gather analysis report
│       │       └── event-filter.html
│       └── openshift-configure-cincinnati/
```

**Where to look by failure type:**

| Symptom | Check these artifacts |
|---|---|
| Test assertion failure | `e2e/build-log.txt` + `junit_e2e_*.xml` |
| OLS service error/crash | `{provider}/cluster/podlogs/lightspeed-app-server-*.log` |
| Postgres issues | `{provider}/cluster/podlogs/lightspeed-postgres-server-*.log` |
| Deployment failure | `{provider}/cluster/pods.yaml` + `deployments.yaml` |
| Image build failure | `build-logs/*.log` |
| Cluster infra issue | `gather-must-gather/artifacts/camgi.html` + `event-filter.html` |
| CI namespace issues | `build-resources/events.json` + `pods.json` |

#### 4d. Downloading artifacts locally

When you need to search across many files or the artifacts are too large
for WebFetch, download them to a temp directory using `gsutil` or `gcloud storage`:

```bash
TMPDIR=$(mktemp -d)
# Download a specific subdirectory
gcloud storage cp -r \
  gs://test-platform-results/pr-logs/pull/{org}_{repo}/{pr}/{job_name}/{build_id}/artifacts/e2e-ols-cluster/e2e/artifacts/ \
  "$TMPDIR/"
```

The GCS bucket path mirrors the Prow URL: strip `https://prow.ci.openshift.org/view/gs/`
and prepend `gs://`.

When multiple jobs have failed, investigate each in a separate subagent (Task tool)
to keep build-log context isolated and run fetches in parallel.

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
