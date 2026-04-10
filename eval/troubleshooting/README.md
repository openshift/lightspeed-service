# OLS Troubleshoot Evals

Evaluation scenarios for the OpenShift LightSpeed troubleshooting agent. Each scenario deploys a broken or misconfigured workload on a live cluster, asks the agent to diagnose the issue, and scores the response against an expected answer using LLM-based metrics.

Scenarios require `oc` access to a running OpenShift cluster. Each one has a `setup.sh` (deploys the faulty resources), `verify.sh` (checks cluster state), and `cleanup.sh` (tears everything down).

## Scenarios

| Tag | Category | Description |
|-----|----------|-------------|
| `envvar_missing` | CrashLoop | Deployment crashes because the `DEPLOY_ENV` environment variable is undefined |
| `batch_failure` | Job Failure | Job `inventory-sync-validator` repeatedly fails to connect to a database |
| `storage_binding` | PVC Misconfiguration | Memcached pod stuck because its PersistentVolumeClaim cannot be provisioned |
| `namespace_pod_count` | Namespace Matching | Count pods in `fleet-alpha` without accidentally including `fleet-alpha1` |
| `scheduled_outage_detection` | Log Analysis | Detect a time-window outage in `report-generator` caused by an external API being unavailable |
| `periodic_failure_window` | Log Analysis | Same as above but the agent must report the exact failure time window (03:00-03:05) |
| `config_drift_analysis` | Log Analysis | Connection refused errors caused by a config reload that loaded staging settings in production |
| `readiness_probe_diagnosis` | Pod Naming Ambiguity | Pod named `catalog-index-service` fails readiness probes; agent must not confuse it with a Kubernetes Service |
| `ingress_rule_mismatch` | NetworkPolicy | `web-portal` times out reaching `api-gateway` because the NetworkPolicy only allows `tier=backend` ingress |
| `oom` | OOMKill | Pod in CrashLoopBackOff due to a deliberate memory leak exceeding its 60Mi limit |
| `networkpolicy` | NetworkPolicy (multi-turn) | Three-turn conversation: diagnose frontend-backend connectivity, identify the blocking NetworkPolicy, and propose a fix |

## Running

```bash
# Single scenario
make oom

# All scenarios
make all
```

