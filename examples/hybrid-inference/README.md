# Hybrid inference example

Two KServe `InferenceService`s behind one `OLSConfig`, exposed as two OLS providers:

| OLS provider / model | Model | ServingRuntime | Image | Hardware |
|---|---|---|---|---|
| `cpu-small` | `ibm-granite/granite-4.2-3b-fp8` | `cpu-small-runtime` | `registry.redhat.io/rhaii/vllm-cpu-rhel9:3.5` | CPU |
| `gpu-large` | `ibm-granite/granite-4.2-8b-fp8` | `gpu-large-runtime` | `registry.redhat.io/rhaii/vllm-cuda-rhel9:3.5` | 1 GPU |

Each InferenceService sets `--served-model-name`, so `/v1/models` returns
`cpu-small` / `gpu-large` and OLS sends those names upstream.

## Layout

- **ServingRuntime**: reusable engine config (image, command, `--port`,
  `--download-dir`, ports, model-cache mount). No model name, no compute requests.
- **InferenceService**: model ID, served name, `--max-model-len`, tool-call flags, and CPU/GPU resources.

## Credentials

- No Hugging Face token: both models are public.
- No vLLM API key: requests are unauthenticated.
- `llmcreds-cpu-small` and `llmcreds-gpu-large` each hold a dummy `apitoken=unused`.
  `OLSConfig` requires `credentialsSecretRef` on every provider; vLLM ignores the value.
- There is **one secret per provider**. The operator mounts each provider's secret at
  `/etc/apikeys/<secret-name>`, so two providers sharing one secret collide on the
  volume name and mount path, and the `lightspeed-app-server` deployment is rejected
  (`ApiReady: False`). `credentialKey` does not avoid this — it only selects a key
  inside the secret.

## Prerequisites

- Lightspeed operator in `openshift-lightspeed`.
- RHOAI/KServe with `RawDeployment` support.
- RHAII 3.5 images pullable from `registry.redhat.io`.
- A node labeled `inference=cpu-small` for `cpu-small`, and a node labeled
  `inference=gpu-large` with one GPU for `gpu-large`. The manifests also tolerate an
  `inference=<value>:NoSchedule` taint on those nodes.
- 16 GiB memory requested (24 GiB limit) per model; the CPU model needs roughly two
  cores' worth of headroom for decoding.
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

- **FP8 weights on CPU.** `granite-4.2-3b-fp8` is served on CPU. If it fails to
  load, check the vLLM log for an unsupported architecture or quantization error, then
  fall back to a non-FP8 model or a CPU-friendly quantization.
- **GPU sizing.** ~8 GB of FP8 weights plus KV cache for 32768 tokens fits on one
  GPU; the request includes `nvidia.com/gpu: 1`. Add `--tensor-parallel-size` and
  raise the GPU request if you move to a larger model.
- **Tool calling.** Both servers run with `--enable-auto-tool-choice
  --tool-call-parser=granite4`. OLS always binds the operator's documentation-search
  tool, so every query sends `tools` with `tool_choice: "auto"`; without those flags
  vLLM rejects the request with `400 "auto" tool choice requires
  --enable-auto-tool-choice and --tool-call-parser to be set`. `introspectionEnabled:
  false` only disables the OpenShift MCP tools; set it to `true` to add them.
  `cpu-small`'s token window is tight for tool definitions and results. Docs
  retrieval (RAG) stays at the operator default.
- **Lab configuration.** One replica, no autoscaling, no NetworkPolicy. The PVCs are
  `ReadWriteOnce`, which limits each model to one replica (or one node).
