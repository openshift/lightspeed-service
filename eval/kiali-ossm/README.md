# Kiali / OSSM Evaluation Scenarios

LLM-based evaluation scenarios for the OpenShift LightSpeed (OLS) troubleshooting agent against a live Kiali / Istio / OpenShift Service Mesh cluster.

Scenarios are defined in [`scenario_evals.yaml`](scenario_evals.yaml). Each one optionally deploys a broken workload via a setup script, submits a query to OLS, and scores the response with an LLM judge using the `custom:answer_correctness` metric.

The full evaluation harness, setup guide, and historical results live in the external repo:
**[kiali/troubleshooting-ls-evaluation](https://github.com/kiali/troubleshooting-ls-evaluation)**

Latest evaluation results: **[RESULTS\_OSSM.md](https://github.com/kiali/troubleshooting-ls-evaluation/blob/main/RESULTS_OSSM.md)**

---

## Scenarios

### `check_mesh_status` — With MCP

> Check the status of the mesh and identify any issues.

The agent uses Kiali/OSSM MCP tools to produce a structured Istio mesh health report covering:
control plane (istiod version and health), observability stack (Prometheus, Grafana, Tempo/Jaeger),
and data plane (namespace health with any DEGRADED/UNHEALTHY namespaces called out).
Each finding must cite specific evidence from tool output and include a concrete remediation step.

---

### `check_mesh_status_no_kiali` — Without MCP

> Check the status of the mesh and identify any issues.

Same question as above but evaluated **without** Kiali/OSSM MCP tools. The agent must rely on
Kubernetes-native tools (`namespaces_list`, `resources_list`, `pods_list`, `events_list`,
`pods_log`) to assess control plane, observability stack, data plane sidecar injection, and
Istio config objects. Numbered findings with severity, evidence, and `kubectl` remediations
are expected.

---

### `check_bookinfo_services` — With MCP

> Check my bookinfo namespace services in my servicemesh.

The agent provides a comprehensive health overview of the `bookinfo` namespace: overall namespace
health status, individual service health for all services (details, productpage, ratings, reviews,
istio-ingressgateway), Istio config validity (Gateway, VirtualService), and the traffic graph
showing call paths, mTLS status, and response times.

---

### `check_latency_bookinfo_issue` — With MCP

> Users are reporting that the Bookinfo productpage is occasionally taking 5+ seconds to load, but it doesn't happen on every request.

The agent investigates intermittent latency using Kiali metrics (P95/P99), the traffic graph,
distributed traces, and logs across the call chain (`ingressgateway → productpage → reviews → ratings`).
If no active issue is found it must state that clearly and provide actionable next steps
(trace sampling, timeout/retry policies). If an issue is active it must identify the root cause
and recommend a mitigation.

---

### `fix_bookinfo_routing` — With MCP

> In our Bookinfo app, the product page only ever shows black or no stars. It never shows red stars.

A `reviews` VirtualService has `weight: 0` for `reviews-v3`, so the version that renders red stars
never receives traffic. The agent must inspect workloads and routing config, identify the zero-weight
root cause, patch the VirtualService, and confirm the fix by reporting the updated spec.

Requires `setup_script` / `cleanup_script` (see [`scenarios/fix_bookinfo_routing/`](scenarios/fix_bookinfo_routing/)).

---

### `fix_bookinfo_fault_injection` — With MCP

> Some users are seeing errors on the Bookinfo product page — the ratings service is broken.

A `ratings` VirtualService injects a 100% HTTP 503 abort fault. The agent must find the
`fault.abort` block, confirm the DestinationRule is not contributing, identify the fault injection
as root cause, and offer to remove or zero out the abort rule to restore normal service.

Requires `setup_script` / `cleanup_script` (see [`scenarios/fix_bookinfo_fault_injection/`](scenarios/fix_bookinfo_fault_injection/)).

---

### `troubleshoot_latency_trace` — With MCP

> The Bookinfo product page is loading very slowly. Can you investigate what is causing the latency?

A 3-second fixed delay is injected on the `ratings` VirtualService. The agent must identify the
`fault.delay` block as root cause, name `ratings` as the responsible service, corroborate with
distributed traces, and remove the delay to confirm the fix.

Requires `setup_script` / `cleanup_script` (see [`scenarios/troubleshoot_latency_trace/`](scenarios/troubleshoot_latency_trace/)).

---

## Running the Evaluations

### Prerequisites

- A Kubernetes/OpenShift cluster accessible via `kubectl` / `oc`
- [OpenShift MCP server](https://github.com/openshift/openshift-mcp-server) running with the `ossm` toolset and Kiali URL configured (see [MCP server section](#mcp-server--kialiossm-toolset) below)
- OLS running and pointing to the MCP server (default `http://localhost:8080`)
- `OPENAI_API_KEY` set, or an `openai_api_key.txt` file at the repo root (used by the judge LLM)

### Setup (once)

```bash
cd eval/kiali-ossm

# Install Istio (demo profile) + Kiali + Bookinfo on the cluster
make setup-env

# Install the evaluation CLI into an isolated venv
make setup-lseval
```

`make setup-env` runs the following steps in order:

| Step | Target | What it does |
|---|---|---|
| 1 | `istioctl` | Downloads `istioctl` v`ISTIO_VERSION` into `_output/bin/` and copies addon manifests |
| 2 | `install-istio` | Installs Istio demo profile + Prometheus + Kiali + Jaeger, enables sidecar injection on `default` |
| 3 | `update-kiali` | Patches the Kiali deployment to image `quay.io/kiali/kiali:KIALI_VERSION` |
| 4 | `install-bookinfo` | Creates the `bookinfo` namespace, deploys the Bookinfo app and ingress gateway |

Override versions if needed:

```bash
make setup-env ISTIO_VERSION=1.28.0 KIALI_VERSION=v2.27.0
```

### MCP server — Kiali/OSSM toolset

The scenarios that use Kiali tools require the [OpenShift MCP server](https://github.com/openshift/openshift-mcp-server)
started with the `ossm` toolset enabled. See the full Kiali integration guide at
[openshift-mcp-server/docs/KIALI.md](https://github.com/openshift/openshift-mcp-server/blob/main/docs/KIALI.md).

A reference config is provided at [`mcp-config.toml`](mcp-config.toml):

```toml
toolsets = ["core", "ossm"]

[toolset_configs.kiali]
url = "http://localhost:20001"   # Kiali console URL (port-forward or route)
# insecure = true               # for self-signed certs in non-production clusters
# certificate_authority = "/path/to/ca.crt"
```

**Start the MCP server** (pointing to Kiali, using your cluster credentials):

```bash
# Port-forward Kiali first if running locally
kubectl -n istio-system port-forward svc/kiali 20001:20001 &

# Start the MCP server with the Kiali toolset
npx @openshift/openshift-mcp-server@latest --config eval/kiali-ossm/mcp-config.toml
```

The server listens on port `8081` by default (configurable). OLS must be running and pointing to it — see [`olsconfig/olsconfig-openai.yaml`](olsconfig/olsconfig-openai.yaml) for the reference OLS config.

> **Note:** The `check_mesh_status_no_kiali` scenario intentionally runs *without* the Kiali
> toolset to benchmark what the agent can achieve using only Kubernetes-native tools.

### Run all scenarios

```bash
make all
```

### Run a single scenario

```bash
make check_mesh_status
make check_mesh_status_no_kiali
make check_bookinfo_services
make check_latency_bookinfo_issue
make fix_bookinfo_routing
make fix_bookinfo_fault_injection
make troubleshoot_latency_trace
```

Results are written to the `results/` directory as CSV and JSON files.

For the full evaluation harness (MCP server setup, OLS container, result dashboard), see
[kiali/troubleshooting-ls-evaluation](https://github.com/kiali/troubleshooting-ls-evaluation)
and its [OSSM.md](https://github.com/kiali/troubleshooting-ls-evaluation/blob/main/OSSM.md) guide.

---

## OLS Configuration

The [`olsconfig/olsconfig-openai.yaml`](olsconfig/olsconfig-openai.yaml) file is the reference
OLS configuration for these evaluations. It connects to:

- **LLM provider** — OpenAI (`gpt-5.4-mini` default, several models available)
- **MCP server** — Kubernetes MCP server at `http://127.0.0.1:8081/mcp` with a
  `kubernetes-authorization` header sourced from the user's k8s token
- **Auth module** — `noop-with-token` (passes the bearer token through without k8s RBAC checks)

---

## Results

Latest results are published at:
**[https://github.com/kiali/troubleshooting-ls-evaluation/blob/main/RESULTS_OSSM.md](https://github.com/kiali/troubleshooting-ls-evaluation/blob/main/RESULTS_OSSM.md)**

The report is generated automatically by `make generate-ossm-results` in the harness repo and
covers pass rate and mean `custom:answer_correctness` score per scenario, broken down by
MCP vs. no-MCP variants and by OLS model.
