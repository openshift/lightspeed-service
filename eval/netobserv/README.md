# OLS NetObserv Evals

Evaluation scenarios for network observability troubleshooting with NetObserv. Each scenario deploys synthetic traffic on a live OpenShift cluster, asks the agent to investigate using NetObserv metrics and/or flow logs, and scores the response with LLM-based metrics.

## Prerequisites

- OLS dev environment per [main README](../../README.md#installation) (`make install-deps`; Python 3.12 via `uv`)
- Eval CLI: `make setup-lseval` in this directory (isolated `.lseval-venv`; installs `lightspeed-evaluation` from the Makefile-pinned git commit `32c80e1`, not the repo `lseval` extra v0.4.0)
- OpenShift cluster with **NetObserv** installed (`FlowCollector` named `cluster`, Ready)
- Relevant eBPF features enabled per scenario (DNSTracking, PacketDrop, NetworkEvents, FlowRTT, etc.)
- OLS running with **obs-mcp** (and optionally kubernetes-mcp-server `netobserv` tools)
- `oc login` to the cluster
- OLS started from **repo root** with an olsconfig that enables **`ols_config.skills`** (all files under `eval/netobserv/olsconfig/` include this). Without skills, the agent will not load the NetObserv investigation workflow.

## Scenarios

| Tag | Category | Description |
|-----|----------|-------------|
| `dns_latency` | DNS | Slow DNS lookups — `netobserv_namespace_dns_latency_seconds`, `DnsLatencyMs` |
| `dns_nxdomain` | DNS | DNS failures / NXDOMAIN — `DNSErrors`, `DNSNxDomain`, flow DNS fields |
| `packet_drops_kernel` | Drops | Kernel packet drops — `PacketDropsByKernel`, `PktDrop*` fields |
| `packet_drops_policy` | Drops | NetworkPolicy denials — `NetpolDenied`, `network_policy_events` |
| `tls_issues` | Ingress/TLS | HTTPS/TLS errors — `Ingress5xxErrors`, `IPsecErrors` |
| `tcp_rtt` | Latency | Slow TCP RTT — `LatencyHighTrend`, `netobserv_namespace_rtt_seconds` |

Each scenario has `scenarios/<tag>/setup.sh` (deploy workloads), `cleanup.sh` (teardown), and `fixtures/manifest.yaml`. Setup waits for workload traffic, then **`wait_for_netobserv_warmup`** (default **120s**) so flows/metrics reach Prometheus/Loki before OLS is queried. Override with `NETOBSERV_WARMUP_SECS=180` if your cluster is slow.

Skills for the agent live at repo root: `skills/netobserv-metrics/` and `skills/netobserv-flow-logs/`. They are injected via skills RAG only when `ols_config.skills` is present in the active olsconfig — **restart OLS** after skill or olsconfig changes.

## Running

```bash
cd eval/netobserv
make setup-lseval   # once
oc login ...
# Judge + agent: OpenAI — export OPENAI_API_KEY or ../../openai_api_key.txt at repo root
```

Start **obs-mcp** (separate terminal, from the obs-mcp repo):

```bash
oc login …
TOOLSETS=metrics,logs LOKI_URL=http://127.0.0.1:8080 make run   # after Loki port-forward if needed
```

Start **OLS** (provider must match `api.provider` in `system.yaml`):

```bash
OLS_CONFIG_FILE=eval/netobserv/olsconfig/olsconfig-openai.yaml uv run make run
```

Run evals (`lightspeed-eval` runs each scenario's `setup_script` before calling OLS — no separate setup step):

```bash
cd eval/netobserv

# Single scenario
make dns_nxdomain

# All scenarios (one run → single evaluation_*_detailed.csv in results/)
make all
```

Debug a failing setup script (prints rollout/log wait errors):

```bash
./scenarios/dns_nxdomain/setup.sh
oc logs -n netobserv-eval-dns-nxdomain -l app=dns-nxdomain-prober --tail=30

# Stale namespace after manifest changes — setup recreates by default; or delete manually:
oc delete namespace netobserv-eval-dns-nxdomain --ignore-not-found --wait=false
./scenarios/dns_nxdomain/setup.sh

# OpenShift restricted SCC: busybox/iperf/python UIDs are patched from the namespace
# openshift.io/sa.scc.uid-range after apply. Set NETOBSERV_EVAL_RECREATE_NS=false to skip
# namespace delete on re-run.
```

## Metrics

Each turn is scored with:

- `custom:answer_correctness` — alignment with `expected_response` in `scenario_evals.yaml` (stricter than GEval)
- `geval:generic_troubleshooting_experience` — evidence-based NetObserv investigation (see `system.yaml`)

`contexts` in `scenario_evals.yaml` are **judge-only** rubrics for GEval; they are not sent to OLS. Investigation recipes and PromQL belong in `skills/`; ground truth for answer correctness stays in `expected_response`.

### Improving `answer_correctness`

GEval often scores higher because it rewards any real NetObserv data. **Answer correctness** compares against `expected_response` in `scenario_evals.yaml` and penalizes incomplete investigations.

| Common gap | Fix |
|------------|-----|
| Stops after alerts or one generic metric | Run the 3-layer workflow in `skills/netobserv-metrics` §7: alerts → domain metrics → flow logs |
| Wrong namespace in PromQL | Always use the namespace from the user query (e.g. `netobserv-eval-drops-kernel`, not `netobserv`) |
| Policy scenario uses `drop_packets_total` only | Use `network_policy_events_total{action="drop"}` + flows with `packetLoss=dropped`; cite **`OVS_DROP_EXPLICIT`** on OpenShift |
| DNS NXDOMAIN with “no metrics” | Enable `ols_config.skills` and restart OLS; NXDOMAIN is **return traffic** — use `DstK8S_Namespace="…"` (not `SrcK8S_Namespace` first); not `{namespace="…"}`; then Loki flows with `DnsName~netobserv-eval.invalid` |
| TCP RTT without metric names | Cite `netobserv_namespace_rtt_seconds` + `histogram_quantile`; use `TimeFlowRttNs` in flows |

After updating skills, restart OLS so the skills RAG index reloads, then re-run `make all`.

## OLS configs

Example OLS configs for eval runs are under `olsconfig/`:

- `olsconfig-openai.yaml` — OpenAI `gpt-4o-mini`; default for `system.yaml`
- `olsconfig-watsonx.yaml` — WatsonX Granite
- `olsconfig-rhoai-vllm.yaml` — RHOAI vLLM endpoint (set `url` to your serving endpoint)

## Troubleshooting

### `500` / `incomplete chunked read` from the LLM backend

Some OpenAI-compatible gateways drop long streaming responses during MCP tool-calling (large tool schemas). Mitigations:

1. **Run obs-mcp** with NetObserv-relevant toolsets only, e.g. `metrics,logs` — avoid loading every MCP tool on the same OLS instance.
2. **Raise MCP timeout** in your olsconfig (e.g. `timeout: 120` on `mcp_servers`) and lower `tool_budget_ratio` if the provider supports it.
3. **Check OLS logs** (`app_log_level: debug` temporarily) for the failing LLM round.
4. **Fallback providers** — `olsconfig-watsonx.yaml` or `olsconfig-openai.yaml` often handle large tool contexts more reliably; update `api.provider` / `api.model` in `system.yaml` to match.

### `422` / provider not valid

`api.provider` in `system.yaml` must match `llm_providers[].name` in the active olsconfig (e.g. `openai`, `watsonx`).

### `[Confident AI Metric Data Log] Invalid API key`

Harmless noise from DeepEval’s optional Confident AI telemetry upload — not the OpenAI judge key. GEval still runs if you see progress like `GEval Turn Metric` and `5/6`. `system.yaml` sets `CONFIDENT_METRIC_LOGGING_ENABLED=0` to silence it. If the judge itself fails, you get LiteLLM/OpenAI auth errors instead; ensure `OPENAI_API_KEY` or `openai_api_key.txt` is set and `unset OPENAI_API_BASE` when using the public OpenAI API.
