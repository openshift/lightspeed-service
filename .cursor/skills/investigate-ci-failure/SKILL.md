---
name: investigate-ci-failure
description: Investigate CI/Prow/Konflux job failures on a GitHub pull request. Use when the user pastes a PR URL and asks about CI failures, red checks, test failures, or wants to understand why a job failed.
disable-model-invocation: true
---

# Investigate CI Failure

Given a PR URL (e.g. `https://github.com/openshift/lightspeed-service/pull/2825`), diagnose why CI jobs failed.

This skill covers two CI systems:
- **Prow** (ci/prow/*) — test and build jobs with artifacts in GCS
- **Konflux** (Red Hat Konflux/*) — supply-chain security pipeline (build, scan, sign, EC)

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
# All checks at a glance (both Prow and Konflux)
gh pr checks {pr} --repo {org}/{repo}

# Prow failures — commit statuses with Prow URLs (use head SHA from step 1)
gh api repos/{org}/{repo}/statuses/{head_sha} \
  --jq '.[] | select(.state == "failure" or .state == "error") | {context, state, target_url}'

# Konflux failures — check runs (different API)
gh api "repos/{org}/{repo}/commits/{head_sha}/check-runs" \
  --jq '.check_runs[] | select(.conclusion == "failure" or .conclusion == "neutral") | {id, name, conclusion, output_title: .output.title}'
```

This gives you failed Prow jobs (with GCS URLs) and failed Konflux checks
(with check run IDs for drill-down). Route Prow failures to section 3–4,
Konflux failures to the Konflux Failures section.

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
5. **Recommended action** — one of:
   - **Retry** — for infra flakes, transient timeouts, EaaS cluster
     issues. Prefer the CI-native retry (Prow rerun button, Konflux
     re-run in UI) over `/retest` PR comment. Ask the user whether
     they want you to post `/retest {job}` to the PR automatically
   - **Fix in PR** — for test failures, scan findings, build errors caused
     by PR changes (include what to fix)
   - **Escalate** — for persistent infra issues, platform-side problems,
     or failures unrelated to the PR that don't resolve on retry

## Konflux Failures

Konflux runs as GitHub check runs (not commit statuses like Prow). All data
is accessible via the GitHub Check Runs API — no Konflux cluster auth needed.

### K1. List Konflux check runs

```bash
gh api "repos/{org}/{repo}/commits/{head_sha}/check-runs" \
  --jq '.check_runs[] | select(.name | test("Konflux"; "i")) | {id, name, status, conclusion, output_title: .output.title}'
```

This returns all Konflux check runs for the commit. Look for `conclusion`
values: `success`, `failure`, `neutral` (warning), `skipped`.

### K2. Get failure details

For each failed or neutral check run, fetch the full output summary:

```bash
gh api repos/{org}/{repo}/check-runs/{check_run_id} \
  --jq '{name, conclusion, output_title: .output.title, output_summary: .output.summary}'
```

The `output.summary` contains a markdown table of all Tekton tasks with
status indicators and links to Konflux UI logs for each task:

```
| Status       | Duration   | Name              |
| ------------ | ---------- | ----------------- |
| 🟢 Succeeded | 19 minutes | build-images      |
| 🔴 Failed    | 1 minute   | clair-scan        |
| 🟢 Succeeded | 1 minute   | sast-snyk-check   |
```

Parse the summary to identify which task(s) failed.

### K3. Check for annotations

Failed check runs may include inline annotations with specific details:

```bash
gh api repos/{org}/{repo}/check-runs/{check_run_id}/annotations \
  --jq '.[] | {path, annotation_level, message}'
```

### K4. Triage by task name

Task names vary by pipeline. Common ones seen across repos:

**Build & scan pipeline**:

| Failed task | What it means | Likely cause |
|---|---|---|
| `build-images` | Container image build failed | Dockerfile error, dependency issue, build timeout |
| `clair-scan` | CVE vulnerability found in image | New dependency introduced a known CVE |
| `clamav-scan` | Malware scan failed | Rare; usually a false positive |
| `sast-snyk-check` | Static analysis security finding | Code pattern flagged by Snyk |
| `sast-shell-check` | Shell script linting failure | Shellcheck error in scripts |
| `sast-unicode-check` | Suspicious Unicode characters | Homoglyph or bidirectional text detected |
| `ecosystem-cert-preflight-checks` | Red Hat certification preflight failed | Image metadata, labels, or structure issue |
| `deprecated-base-image-check` | Base image is deprecated | Update base image in Dockerfile |
| `rpms-signature-scan` | RPM signature verification failed | Unsigned or tampered RPM in image |
| `prefetch-dependencies` | Dependency prefetch failed | Network issue or invalid dependency reference |

**Integration test pipeline**:

| Failed task | What it means | Likely cause |
|---|---|---|
| `eaas-provision-space` | EaaS namespace provisioning failed | Infra issue, quota exhausted |
| `provision-cluster` | Test cluster provisioning failed | Infra issue, cluster pool exhausted |
| `ols-install` | OLS operator installation failed | Operator or CRD issue in the PR changes |
| `ols-operator-tests` | e2e operator tests failed or timed out | Test failure or timeout (2h limit) — check artifacts on Quay |
| `export-logs-for-retention` | Log export to Quay failed | Usually infra; artifacts may be missing |

**Separate check**:

| Check | What it means | Likely cause |
|---|---|---|
| Enterprise Contract | Policy violation on built image | Missing signatures, provenance, or policy rules not met |

This list is not exhaustive — new tasks may appear. Use the task name and
the check summary (K2) to understand what failed.

### K5. Cross-reference with PR changes

| Failure type | Likely cause |
|---|---|
| `clair-scan` after dependency bump | New dependency has a CVE — check `pyproject.toml` / `uv.lock` changes |
| `build-images` failure | Dockerfile change or new dependency that breaks the build |
| `sast-*` failure | New code pattern flagged by static analysis |
| Enterprise Contract warning | Usually not blocking for PRs (runs on merge) — check if `conclusion` is `neutral` vs `failure` |
| All tasks pass but check is `neutral` | EC ran as optional/warning — informational, not blocking |

### K6. Fetching artifacts from Quay (no auth needed)

Konflux uploads scan results as OCI attachments on the built image in
Quay. These are **publicly accessible** — no Konflux auth required.

The Quay image path is Konflux-configured and varies per repo:

| Repo | Quay image base |
|---|---|
| `openshift/lightspeed-service` | `quay.io/redhat-user-workloads/crt-nshift-lightspeed-tenant/ols/lightspeed-service` |
| `openshift/lightspeed-operator` | `quay.io/openshift-lightspeed/ols-operator-artifacts` |

The `head_sha` comes from step 1 (`gh api .../pulls/{pr}`). Tag patterns
vary per repo — check the STEP-UPLOAD log or try common patterns:

| Tag pattern | Example |
|---|---|
| `on-pr-{head_sha}` | lightspeed-service multi-arch index |
| `on-pr-{head_sha}-linux-x86-64` | lightspeed-service arch-specific image |
| `{head_sha}` | lightspeed-operator artifacts |

This path is fully automated — no browser, no user input, no cluster auth.

**Step 1**: List attachments on the image (try the main tag first):
```bash
oras discover quay.io/{image_base}:{tag}
```

**Step 2**: If arch-specific images exist, check those too (Clair, SBOM):
```bash
oras discover quay.io/{image_base}:{tag}-linux-x86-64
```

Attachment types:

| Media type | File | Contents |
|---|---|---|
| `application/sarif+json` | `shellcheck-results.sarif` | ShellCheck SAST findings |
| `application/sarif+json` | `sast_snyk_check_out.sarif` | Snyk SAST findings |
| `application/sarif+json` | `sast_unicode_check_out.sarif` | Unicode control char findings |
| `application/vnd.clamav` | ClamAV result | Malware scan findings |
| `application/vnd.redhat.clair-report+json` | `clair-report-amd64.json` | Full Clair CVE report (on arch images) |
| SBOM (cosign attachment) | `sbom.json` | SPDX SBOM (tag: `sha256-{digest}.sbom`) |

**Step 3**: Fetch a specific attachment by digest:
```bash
TMPDIR=$(mktemp -d)
oras pull -o "$TMPDIR" quay.io/{image_base}@sha256:{digest}
```

**Step 4**: Parse results.

For SARIF files (SAST scans):
```bash
python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
for run in d.get('runs', []):
    tool = run['tool']['driver']['name']
    results = run.get('results', [])
    print(f'{tool}: {len(results)} finding(s)')
    for r in results:
        print(f'  [{r[\"level\"]}] {r[\"message\"][\"text\"][:200]}')
" "$TMPDIR"/*.sarif
```

For Clair reports (CVE scan):
```bash
python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
vulns = d.get('vulnerabilities', {})
pkg_vulns = d.get('package_vulnerabilities', {})
print(f'Total vulnerabilities: {len(vulns)}')
by_sev = {}
for vid, v in vulns.items():
    sev = v.get('normalized_severity', 'Unknown')
    by_sev.setdefault(sev, []).append(v)
for sev in ['Critical', 'High', 'Medium', 'Low', 'Unknown']:
    if sev in by_sev:
        print(f'  {sev}: {len(by_sev[sev])}')
" "$TMPDIR"/clair-report-*.json
```

This is the most reliable path for Konflux scan failures — fully automated,
gives the exact findings without needing Konflux UI access or user input.

Artifact content varies — don't assume a fixed structure. After pulling,
inspect what you got and adapt:

```bash
oras pull -o "$TMPDIR" quay.io/{image_base}:{tag}
find "$TMPDIR" -type f | head -30
```

Known artifact types you may encounter:

| What you find | How to parse |
|---|---|
| `*.sarif` files | SARIF JSON — SAST scan results (see parser above) |
| `clair-report-*.json` | Clair CVE report (see parser above) |
| `konflux-artifacts/` directory | Cluster state dumps (JSON: clusteroperators, events, RBAC, CSVs, etc.) |
| `openai/`, `azure_openai/`, etc. | Per-provider e2e test results |
| `leaktk-scan-*.log` | Leak detection scan logs (plain text) |
| `sbom.json` / `*.sbom` | SPDX SBOM |
| ClamAV results | Malware scan findings |

Explore the content, identify the relevant files, and parse accordingly.

**Limitations** — fall back to K7 (ask user for Konflux UI logs) when:
- **No artifacts on Quay** — some tasks don't upload results (e.g., build
  failures, infra steps)
- **Task timeouts** (`TaskRunTimeout`) — the task was killed before it
  could upload anything. Identifiable from the check run summary (K2):
  `❓ Reason: TaskRunTimeout`. Ask the user to open the Konflux UI log
  link for the timed-out task and paste the output

### K7. Fallback: ask user for Konflux UI logs

Use this only when K6 (Quay artifacts) doesn't have what you need — no
artifacts uploaded, task timed out, or the failure isn't explained by the
scan results.

The full task logs are only available through the **Konflux UI** in a
browser (SSO-authenticated). The Tekton Results API is not externally
accessible, and completed PipelineRuns are pruned from the Kubernetes API.

Provide the user with the direct Konflux UI log link for the failed task.
The URL pattern is in the check summary markdown (parsed in K2), e.g.:
```
https://konflux-ui.apps.stone-prd-rh01.pg1f.p1.openshiftapps.com/ns/crt-nshift-lightspeed-tenant/pipelinerun/<pipelinerun-name>/logs/<task-name>
```

Ask the user to open the link and paste the relevant log output into chat
for further analysis.

### K8. Parsing task logs (when user pastes from UI)

Each Konflux task log is divided into steps with `STEP-*` headers.

For **scan tasks**, the structure is:

```
STEP-USE-TRUSTED-ARTIFACT     ← artifact download (noise)
STEP-<ACTUAL-CHECK>            ← the scan step (look here)
STEP-UPLOAD                    ← result upload (noise)
```

For **e2e integration test tasks**, the structure is:

```
STEP-GET-KUBECONFIG            ← cluster credentials (noise unless it fails)
STEP-RUN-E2E-TESTS             ← the actual test run (look here)
STEP-PUSH-ARTIFACTS            ← upload results to Quay (noise)
```

When the user pastes a task log:

1. **Skip** infra steps (`STEP-USE-TRUSTED-ARTIFACT`, `STEP-UPLOAD`,
   `STEP-GET-KUBECONFIG`, `STEP-PUSH-ARTIFACTS`)
2. **Focus on the main step** — this is where the actual work happens
3. **E2e logs are extremely repetitive** — deployment readiness polls
   repeat every 5 seconds for 15+ minutes per test retry. Deduplicate
   mentally: look for the first occurrence of a failure pattern, then
   skip to the next `[FAILED]` or `[PANICKED]` marker. Key signals:
   - `[FAILED]` / `[PANICKED]` markers with file:line references
   - `Unexpected error:` blocks with the actual error message
   - Transitions like `node.kubernetes.io/unreachable` or
     `no nodes available to schedule pods` (infra failure)
   - `make: *** ... Terminated` (task was killed by timeout)
   - Ginkgo test summary: `X failed, Y passed, Z skipped`
4. **For scan failures**, look for the `TEST_OUTPUT` JSON:
   ```json
   {"result":"SUCCESS","note":"Task ... success: No finding was detected",...}
   ```
   or:
   ```json
   {"result":"FAILURE","note":"Task ... failure: <N> findings detected",...}
   ```

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
| `Red Hat Konflux / lightspeed-service-on-pull-request` | Konflux PR pipeline — build, scan (Clair, ClamAV, Snyk, ShellCheck, Unicode), preflight, SBOM |
| `Red Hat Konflux / ols-enterprise-contract / lightspeed-service` | Enterprise Contract policy check — signatures, provenance, CVEs. Often `neutral` (warning) on PRs, blocking on merge. |

Other repos may have additional Prow and Konflux jobs not listed here.
Use `gh pr checks` and the APIs in step 2 to discover them.

## Tool usage notes

- **`gh`** — all GitHub API calls (PR metadata, statuses, checks, comments, files).
- **`oras`** — fetch Konflux scan results and artifacts from Quay (OCI attachments). Required for K6.
- **`WebFetch`** — Prow artifacts from GCS (`gcsweb-ci.apps.ci.l2s4.p1.openshiftapps.com/gcs/...` for browsing, `storage.googleapis.com/test-platform-results/...` for raw content). The Prow dashboard URL itself is JS-rendered and not useful via WebFetch — always use GCS URLs.
- **Prow** failures appear as **commit statuses** (`gh api repos/.../statuses/{sha}`). **Konflux** failures appear as **check runs** (`gh api repos/.../check-runs/{id}`). Use both APIs when listing failures.
- **Konflux triage order**: (1) `gh api` check run summary for task-level pass/fail, (2) `oras` to fetch scan results and artifacts from Quay, (3) ask user for Konflux UI logs only as a last resort.
- Build logs can be very large. When fetched via WebFetch, they're saved to a temp file — read from the end to find failures quickly.
