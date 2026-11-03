# Hybrid inference example

Two KServe `InferenceService`s behind one `OLSConfig`, exposed as two OLS providers:

| OLS provider / model | Model | ServingRuntime | Image | Hardware |
|---|---|---|---|---|
| `cpu-small` | `Qwen/Qwen-AgentWorld-35B-A3B` | `cpu-small-runtime` | `registry.redhat.io/rhaii/vllm-cpu-rhel9:3.5` | CPU |
| `gpu-large` | `RedHatAI/gemma-4-31B-it-FP8-block` | `gpu-large-runtime` | `registry.redhat.io/rhaii/vllm-cuda-rhel9:3.5` | 4 GPUs, TP=4 |

Each InferenceService sets `--served-model-name`, so `/v1/models` returns
`cpu-small` / `gpu-large` and OLS sends those names upstream.

## Layout

- **ServingRuntime**: reusable engine config (image, command, `--port`,
  `--download-dir`, ports, model-cache mount). No model name, no compute requests.
- **InferenceService**: model ID, served name, model flags (`--max-model-len`,
  `--tensor-parallel-size`, parsers, `--language-model-only`) and CPU/GPU resources.

## Credentials

- No Hugging Face token: both models are public.
- No vLLM API key: requests are unauthenticated.
- `llmcreds` holds a dummy `apitoken=unused`. `OLSConfig` requires
  `credentialsSecretRef` on every provider; vLLM ignores the value.

## Prerequisites

- Lightspeed operator in `openshift-lightspeed`.
- RHOAI/KServe with `RawDeployment` support.
- RHAII 3.5 images pullable from `registry.redhat.io`.
- A node with 4 GPUs for `gpu-large`.
- Enough CPU/memory for `cpu-small` (35B MoE, ~70 GB of bf16 weights).
- Network access to Hugging Face for the first start. Weights are cached on one PVC
  per model (`cpu-small-models` 100Gi, `gpu-large-models` 50Gi) using the cluster's
  default StorageClass, so later restarts skip the download.

## Apply and verify

```bash
oc apply -k examples/hybrid-inference

oc get servingruntime,inferenceservice -n openshift-lightspeed
oc get pods -n openshift-lightspeed -l serving.kserve.io/inferenceservice

oc run curl-test --rm -it --restart=Never -n openshift-lightspeed \
  --image=curlimages/curl -- \
  curl -s http://gpu-large-predictor.openshift-lightspeed.svc.cluster.local:8080/v1/models
```

Query through OLS:

```bash
curl -sk -H "Authorization: Bearer $(oc whoami -t)" \
  -H 'Content-Type: application/json' \
  "https://<ols-route>/v1/query" \
  -d '{"query":"what is a deployment?","provider":"gpu-large","model":"gpu-large"}'
```

## Routing by mode

OLS does not map `mode` to a provider/model; `mode` only selects the system prompt.
To send ASK traffic to `cpu-small` and TROUBLESHOOTING traffic to `gpu-large`, have
the client send `provider`/`model` per request, or put a proxy in front of OLS.

## Caveats

- **FP8 hardware.** The Gemma model uses block-wise FP8, which generally targets
  Ada/Hopper-class GPUs. Ampere cards (e.g. A10G) may reject it or run slowly.
- **GPU sizing.** ~31 GB of FP8 weights. TP=4 is chosen for headroom; TP=2 may suffice.
- **Qwen on CPU.** `--max-model-len=8192` and 96-128 GiB are sized for bf16 weights
  plus KV cache. If the model fails to load, check the vLLM log for unsupported
  architecture or out-of-memory, then reduce context or use a quantized model.
- **Tool calling.** Both InferenceServices enable `--enable-auto-tool-choice` with a
  model-specific tool parser (`gemma4`, `qwen3_coder`), but the example sets
  `introspectionEnabled: false`, so OLS has no tools to call. Set it to `true` to
  enable the built-in OpenShift MCP server; `cpu-small`'s 8192-token window is tight
  for tool definitions and results. Docs retrieval (RAG) stays at the operator default.
- **Lab configuration.** One replica, no autoscaling, no NetworkPolicy. The PVCs are
  `ReadWriteOnce`, which limits each model to one replica (or one node).
