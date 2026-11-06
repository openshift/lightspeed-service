# Running experiments

Compare models and settings on the hybrid inference stack (OLS, Praxis and a CPU-served model) by
running the same evaluation many ways and comparing the results.

Experiments live in [`eval/experiments.yaml`](../experiments.yaml); a Job runs them one at a time.
Per experiment it patches the `cpu-small` InferenceService (model, parsers, thinking, CPU pinning and
resources), waits for vLLM, checks the CPU binding, runs `lightspeed-eval` with your judges, writes
`run_meta.json` and rebuilds the dashboard. It never touches `gpu-large`.

| File | Purpose |
|---|---|
| `eval/experiments.yaml` | Models, judges and experiments |
| `eval/scripts/run_experiments.py` | The runner |
| `eval/scripts/build_dashboard.py` | Builds the dashboard from a results folder |
| `eval/system_hybrid_ask_streaming_cluster.yaml` | Eval config: OLS address, ask header, metrics, judge template (in the image) |
| `eval/cluster/` | Manifests for running on the cluster |
| `eval/cluster/build/` | Maintainer only: builds the image ([build/README.md](build/README.md)) |

## Prerequisites

- OLS with hybrid routing and Praxis, and the model InferenceServices deployed
  ([examples/hybrid-inference/](../../examples/hybrid-inference/README.md)).
- `oc` logged in with rights to create the objects below, and judge credentials for step 2.
- Gated models: accept the licence on Hugging Face, create the token secret, and set `gated: true` in
  `experiments.yaml` (the plan stops if either is missing):
  ```bash
  oc create secret generic hf-token -n openshift-lightspeed --from-literal=HF_TOKEN="$HF_TOKEN"
  ```

## Quick start

From the repository root.

1. **Once per cluster:** the volume, service account, permissions and a NetworkPolicy that lets the
   Job reach OLS.
   ```bash
   oc apply -f eval/cluster/eval-runner.yaml
   ```

2. **Judge credentials.** `hosted_vllm` needs both values; an `openai` judge also needs
   `OPENAI_API_KEY`.
   ```bash
   oc create secret generic eval-credentials -n openshift-lightspeed \
     --from-literal=HOSTED_VLLM_API_KEY="$HOSTED_VLLM_API_KEY" \
     --from-literal=HOSTED_VLLM_API_BASE="$HOSTED_VLLM_API_BASE"
   ```

3. **Edit `eval/experiments.yaml`**, then publish it and start the Job:
   ```bash
   oc create configmap experiments -n openshift-lightspeed \
     --from-file=experiments.yaml=eval/experiments.yaml \
     --dry-run=client -o yaml | oc apply -f -

   oc create -f eval/cluster/job.yaml
   oc logs -f -n openshift-lightspeed job/<name printed above>
   ```
   The Job mounts the ConfigMap, so experiment edits need no rebuild. Runner arguments go in `args:`
   in `job.yaml`: `--only "My experiment"`, `--rerun`, or remove `--run` to print the plan only.

4. **Copy the results out** after the Job finishes (the volume is ReadWriteOnce, one pod at a time):
   ```bash
   mkdir -p ~/results_experiments
   oc create -f eval/cluster/results-reader.yaml
   oc wait --for=condition=Ready pod/results-reader -n openshift-lightspeed --timeout=180s
   oc rsync -n openshift-lightspeed results-reader:/results/experiments/ ~/results_experiments/
   oc delete pod results-reader -n openshift-lightspeed
   ```
   Open `~/results_experiments/dashboard.html`.

## experiments.yaml

**`models:`** each model once; `model` and `tool_call_parser` are required, `reasoning_parser` is
optional (its flag is not passed when absent). Values come from the model's card.

| Key | Meaning |
|---|---|
| `model` | Hugging Face id vLLM serves |
| `tool_call_parser` | vLLM `--tool-call-parser` (adds `--enable-auto-tool-choice`), or `none` |
| `reasoning_parser` | Optional vLLM `--reasoning-parser` |
| `gated` | `true` to give the pod the `hf-token` secret as `HF_TOKEN` |

**`judges:`** each judge once with `provider` and `model` (optional `temperature`,
`max_completion_tokens`); an experiment names one or more, and every answer is scored by each.

**`defaults:`** shared settings, overridable per experiment.

| Key | Meaning |
|---|---|
| `omp_threads_bind` | vLLM compute CPUs: `"16-22"` pins them, `auto` uses one thread per core minus one |
| `thinking` | On by default; `false` turns it off. Models without an `enable_thinking` switch ignore it (the plan warns when it can tell) |
| `judges` | Judge names |
| `max_model_len` | Context size |
| `cpu_request`, `cpu_limit`, `memory_request`, `memory_limit` | Pod resources; see "Sizing" |
| `served_model_name` | Name OLS and Praxis use (keep `hybrid-model`) |
| `eval_data` | `eval_data_short.yaml` (10 questions) or `eval_data.yaml` (797) |
| `max_conversations` | First N conversations only |
| `extra_args`, `extra_env` | More vLLM arguments and environment variables |
| `dedupe_on` | What counts as the same experiment |

**`experiments:`** a unique `name` (shown in the dashboard), a `model` from `models:`, and anything
that differs from the defaults:
```yaml
experiments:
  - name: "Qwen3.5-4B - 7 compute"
    model: qwen3.5-4b
    omp_threads_bind: "16-22"
  - name: "Qwen3.5-4B - 7 compute, no thinking"
    model: qwen3.5-4b
    omp_threads_bind: "16-22"
    thinking: false
```

A finished run is skipped when it matches on model, exact CPUs, context size, dataset (name, content
and subset), judges, thinking and both parsers; the name is not compared. Starting the Job again
therefore runs only new experiments and resumes a stopped one, while changing settings under the same
name reruns it (with a warning). `--rerun` repeats everything; memory changes are not a new experiment.

## Sizing

On CPU, vLLM reserves **92% of the pod's memory limit** (`--gpu-memory-utilization`) and fills the
rest with KV cache, so the pod always looks nearly full. The limit must:

1. leave that 92% free at startup, with other processes using about 1.7 GiB: use at least **32 Gi**
   for any model;
2. cover the weights, about 3 GiB of overhead and the KV cache: use **checkpoint x 1.25 + 8 GiB**,
   with the request equal to the limit.

The plan checks this against each checkpoint and the node, as advice. The defaults (CPU 24/28, memory
300Gi/300Gi) cover every model in the file; the node has about 495 GiB.

| Model | Checkpoint | Needs |
|---|---|---|
| Granite 4.2 3B | 6.8 GiB | 32 GiB |
| Qwen3.5-4B | 8.7 GiB | 32 GiB |
| Qwen-AgentWorld 35B-A3B | 64.6 GiB | ~89 GiB |

Each model is downloaded once to the `cpu-small-models` volume (500 Gi), so it must hold all of them.
Raise `storage` in `examples/hybrid-inference/cpu-small/pvc.yaml` to grow it: the class expands
online, volumes cannot shrink, and AWS allows one resize per volume every 6 hours.

## What you get

In `/results/experiments`: per run, `eval_<timestamp>/<agent>/run_1/` with the detailed CSV, summary
JSON (each judge's scores), graphs and `run_meta.json` (served model, AWS instance, bound cores,
resources, thinking, parsers, judges, vLLM version, throttling counters); `rendered_manifests/` and
`rendered_configs/` hold the exact manifests and configs used; and `dashboard.html` compares runs one
metric at a time, with filters on run settings and a judge view (recorded, average, max, majority, or
each judge).

## Notes

- **Every experiment restarts the model pod** (about 10 minutes for a 35B model, plus its first
  download).
- **Pod `Pending`:** the node lacks the `inference` label or taint.
- **The eval cannot reach OLS:** the operator's NetworkPolicy admits only the console, Prometheus and
  the ingress controller; `eval-runner.yaml` admits the Job's pods.
- **`ImagePullBackOff`:** the cluster cannot reach `quay.io`.
- **"Missing judge credentials":** add that provider's variables to `eval-credentials`.
- **Gated model fails:** the `hf-token` secret is missing, or the token is not approved for the model
  (`401`/`403` while downloading).
- **Thinking on:** time to first token may be the first answer token rather than the first generated
  one, and throughput includes reasoning time; compare such runs with care.
- **Scores:** the dashboard applies the metric threshold to each saved score, so its pass/fail can
  differ from the eval's own.
