# RFE-9380: In-Process LLM Credential Hot-Reload

## Problem

When `credentialsSecretRef` is updated, the Lightspeed operator triggers a rolling
restart of `lightspeed-app-server`. For customers using short-lived LLM tokens
(e.g. 1-hour validity rotated by CronJob), this causes hourly pod restarts and
temporary capacity loss.

- **RFE:** [RFE-9380](https://redhat.atlassian.net/browse/RFE-9380)
- **Customer environment:** OCP 4.21, pre-production
- **Current workaround:** replicas >= 3 with manual PDB (minAvailable: 2)

## Solution

The service re-reads credential files from disk on every LLM request (Prometheus
`credentials_file` pattern), so rotated secrets are picked up without a pod
restart.

### Why this matters

The app-server pod contains the FAISS vector indexes (~100MB) and the
SentenceTransformer embedding model (~400MB), all loaded into memory at startup.
A rolling restart means:

| Step | Time |
|------|------|
| FAISS index reload | ~5-15s |
| Embedding model reload | ~3-5s |
| Readiness probe pass | ~2-5s |
| **Total cold start** | **~10-25s per pod** |

With hourly rotation and 3 replicas, that is up to 24 restart cycles/day with
degraded capacity during each one.

### Cost comparison

| Factor | Hot-Reload | 2+ Replicas workaround |
|--------|-----------|------------------------|
| CPU per request | 1 file read (~0.05ms) | Zero extra |
| Memory | Zero extra | 2-3x pod memory |
| Compute cost | Negligible | 2-3x vCPU reserved |
| Availability during rotation | 100% | ~95-99% (brief capacity drop) |
| Ops complexity | None | PDB + replica tuning + alert tuning |

## Design

Since `load_llm()` already creates a new `LLMProvider` per request, we add a
`get_credentials()` method to `ProviderConfig` that re-reads the credential file
each time it is called.

```text
Request → load_llm() → LLMProvider.__init__()
                          → provider_config.get_credentials()
                              → open(credentials_path) + read()
                              → return fresh value
```

### Why re-read on every request (vs caching or fsnotify)

- **Kubernetes symlink safety** — kubelet uses atomic symlink swaps (`..data`);
  `os.stat()` mtime can be unreliable after swaps. Re-reading avoids edge cases.
- **Proven pattern** — Prometheus `credentials_file` / `password_file` re-reads
  on every scrape request.
- **No new dependencies** — no watchdog, fsnotify, or background threads.
- **Negligible overhead** — one `open()+read()` of a <100 byte file per LLM
  request is trivial vs multi-second LLM call latency.
- **Thread-safe by default** — each request reads independently.

### Ecosystem alignment

| Project | Pattern |
|---------|---------|
| Prometheus | `credentials_file` re-read on every scrape |
| OpenShift library-go | `fileobserver` hash-based polling (Go operators) |
| controller-runtime | `CertWatcher` fsnotify + periodic re-read |
| kube-proxy | watches parent directory (PR #139204) |
| Stakater Reloader | triggers rolling restart (what we are moving away from) |

## Files Changed

### Core

- `ols/utils/checks.py` — added `read_secret_from_path()` helper
- `ols/app/models/config.py` — added `_credentials_path` private attr and
  `get_credentials()` method to `ProviderConfig`

### Providers (7 files)

All providers changed from `self.provider_config.credentials` to
`self.provider_config.get_credentials()`:

- `ols/src/llms/providers/openai.py`
- `ols/src/llms/providers/azure_openai.py`
- `ols/src/llms/providers/watsonx.py`
- `ols/src/llms/providers/rhoai_vllm.py`
- `ols/src/llms/providers/rhelai_vllm.py`
- `ols/src/llms/providers/google_vertex.py` (both Gemini and Anthropic)
- `ols/src/llms/providers/bedrock.py`

### Tests (9 new tests)

- `tests/unit/utils/test_checks.py` — 4 tests for `read_secret_from_path()`
- `tests/unit/app/models/test_config.py` — 4 tests for `get_credentials()`
- `tests/unit/llms/providers/test_openai.py` — 1 end-to-end rotation test

## Testing on a Cluster

### Local with Podman

```bash
# Build image with changes
OLS_API_IMAGE=localhost/ols:hot-reload make images

# Prepare config
mkdir -p /tmp/ols-test/config
cp examples/olsconfig.yaml /tmp/ols-test/config/olsconfig.yaml
echo "your-api-key" > /tmp/ols-test/config/apikey.txt

# Run
podman run -it --rm \
  -v /tmp/ols-test/config:/app-root/config:Z \
  -e OLS_CONFIG_FILE=/app-root/config/olsconfig.yaml \
  -p 8080:8080 localhost/ols:hot-reload

# Query, then rotate, then query again
curl -X POST http://localhost:8080/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "what is a pod?"}'

echo "new-rotated-key" > /tmp/ols-test/config/apikey.txt

curl -X POST http://localhost:8080/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "what is a deployment?"}'
```

### On OpenShift

```bash
# Build and push
OLS_API_IMAGE=quay.io/<user>/lightspeed-service-api:hot-reload make images
podman push quay.io/<user>/lightspeed-service-api:hot-reload

# Patch existing deployment
oc set image deployment/lightspeed-app-server \
  lightspeed-app-server=quay.io/<user>/lightspeed-service-api:hot-reload \
  -n openshift-lightspeed

# Verify no restarts after secret rotation
oc get pods -n openshift-lightspeed -w
```

### Validation checklist

| Check | How to verify |
|-------|---------------|
| No pod restarts | `oc get pods` — restartCount stays 0 |
| Credential picked up | Queries succeed after rotation |
| No capacity loss | Service responds during rotation window |
| No restart logs | No container start messages in logs |

## Multi-Repo Contribution

This feature spans two repositories:

| PR | Repo | Purpose | Status |
|----|------|---------|--------|
| Service PR (first) | `openshift/lightspeed-service` | Add `get_credentials()` hot-reload | Ready |
| Operator PR (second) | `openshift/lightspeed-operator` | Skip restart on credential-only secret changes | Future |

The service PR is **backward compatible** — if the operator still restarts pods,
nothing breaks. The operator PR depends on the service change being merged first.

Both PRs reference RFE-9380 and cross-link each other.
