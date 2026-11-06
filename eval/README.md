# OpenShift Lightspeed - Evaluation

Evaluate OLS answer quality with
[lightspeed-evaluation](https://github.com/lightspeed-core/lightspeed-evaluation), run as a Job on
the cluster.

## Running evaluations

[`experiments.yaml`](experiments.yaml) lists what to run, and
[`cluster/README.md`](cluster/README.md) has the quick start. The Job patches the `cpu-small`
InferenceService, waits for vLLM, runs `lightspeed-eval` against OLS scored by your judges, records
how each run was served, and builds a dashboard comparing the runs.

## Datasets and configs

| File | Contents |
|---|---|
| `eval_data_short.yaml` | 10 conversations, for smoke tests |
| `eval_data.yaml` | 797 general OpenShift questions |
| `eval_data_cluster_updates.yaml` | 19 cluster-update conversations |

Point `eval_data:` in `experiments.yaml` at any of them.

`system_hybrid_ask_streaming_cluster.yaml` is the config the Job passes to `lightspeed-eval`: OLS's
in-cluster address, the `ask`-mode header so Praxis routes to `cpu-small`, the correctness metric and
a judge template the runner fills in for each experiment.

Cluster-update questions need OLS itself to hold cluster-reader and monitoring-edit rights
(`oc apply -f rbac-cluster-updates.yaml`), or its answers are access-restricted and the quality
checks fail.

## Other checks

- **Daily CI:** the `ols-eval-periodic-daily` periodic runs `tests/scripts/test-lseval-daily.sh` at
  07:15 UTC on the 10-question dataset, publishing pass-rate and score history with its Prow
  artifacts.
- **Tool-result inspection:** a separate harness with labelled cases in `tool_result_inspection/`;
  see [tool_result_inspection/README.md](tool_result_inspection/README.md).
