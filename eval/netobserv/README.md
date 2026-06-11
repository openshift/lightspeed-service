# OLS NetObserv Evals

Evaluation scenarios for network observability troubleshooting with NetObserv. Each scenario deploys synthetic traffic on a live OpenShift cluster, asks the agent to investigate using NetObserv metrics and/or flow logs, and scores the response with LLM-based metrics.

## Prerequisites

- OLS dev environment per [main README](../../README.md#installation) (`make install-deps`; Python 3.12 via `uv`)
- Eval CLI: `make setup-lseval` in this directory (isolated `.lseval-venv`; repo `lseval` extra is v0.4.0 and cannot pass `mode: troubleshooting`)
- OpenShift cluster with **NetObserv** installed (`FlowCollector` named `cluster`, Ready)
- Relevant eBPF features enabled per scenario (DNSTracking, PacketDrop, NetworkEvents, FlowRTT, etc.)
- OLS running with **obs-mcp** (and optionally kubernetes-mcp-server `netobserv` tools)
- `oc login` to the cluster

## Scenarios

| Tag | Category | Description |
|-----|----------|-------------|
| `dns_latency` | DNS | Slow DNS lookups — `netobserv_namespace_dns_latency_seconds`, `DnsLatencyMs` |
| `dns_nxdomain` | DNS | DNS failures / NXDOMAIN — `DNSErrors`, `DNSNxDomain`, flow DNS fields |
| `packet_drops_kernel` | Drops | Kernel packet drops — `PacketDropsByKernel`, `PktDrop*` fields |
| `packet_drops_policy` | Drops | NetworkPolicy denials — `NetpolDenied`, `network_policy_events` |
| `tls_issues` | Ingress/TLS | HTTPS/TLS errors — `Ingress5xxErrors`, `IPsecErrors` |
| `tcp_rtt` | Latency | Slow TCP RTT — `LatencyHighTrend`, `netobserv_namespace_rtt_seconds` |

Each scenario has `scenarios/<tag>/setup.sh` (deploy workloads), `cleanup.sh` (teardown), and `fixtures/manifest.yaml`. Allow a few minutes after setup for NetObserv to export metrics and flows.

Skills for the agent live at repo root: `skills/netobserv-metrics/` and `skills/netobserv-flow-logs/`.

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

# All NetObserv scenarios
make all
```

Debug a failing setup script (prints rollout/log wait errors):

```bash
./scenarios/dns_nxdomain/setup.sh
oc logs -n netobserv-eval-dns-nxdomain -l app=dns-nxdomain-prober --tail=30

# DNS latency — delete stale namespace if the prober image changed (ubi → busybox)
oc delete namespace netobserv-eval-dns-latency --ignore-not-found --wait=false
./scenarios/dns_latency/setup.sh

# TLS — requires openssl on the host; setup.sh generates a self-signed cert (not in git)
oc delete namespace netobserv-eval-tls --ignore-not-found --wait=false
./scenarios/tls_issues/setup.sh
```

## Metrics

Each turn is scored with:

- `custom:answer_correctness` — alignment with the expected investigation outcome
- `geval:generic_troubleshooting_experience` — evidence-based NetObserv investigation (see `system.yaml`)

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
