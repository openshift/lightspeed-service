# RHOAI Test Infrastructure

Provisions Red Hat OpenShift AI (RHOAI) operators, GPU infrastructure, and a
vLLM model-serving endpoint so the LSEval periodic suite can run against a
self-hosted Llama-3.1-8B-Instruct model instead of a hosted API.

## When to use

Set `RHOAI_PROVISION=true` in the CI job environment (or export it locally)
before running `tests/scripts/test-lseval-periodic.sh`. When the flag is unset
or `false`, the script falls back to the default OpenAI GPT-4o-mini path.

## Required environment variables

| Variable | Purpose |
|---|---|
| `RHOAI_PROVISION` | Set to `true` to enable this flow |
| `HUGGING_FACE_HUB_TOKEN` | HuggingFace token to download model weights. Currently used for Llama 3.1 8B but works with any HuggingFace-hosted model. Larger models may require scaling GPU nodes (more GPUs, higher VRAM) |
| `VLLM_API_KEY` | Arbitrary value — we define it ourselves. The same value is set as the vLLM endpoint secret and passed to OLS as the provider key |
| `OPENAI_PROVIDER_KEY_PATH` | Path to file containing the OpenAI API key (for the judge LLM) |
| `OLS_IMAGE` | OLS container image pullspec to deploy |

## Cluster prerequisites

- OpenShift 4.x cluster with GPU-capable nodes (e.g. AWS `g4dn`, `g5`, `p3`, `p4` instance types).
- OLM (Operator Lifecycle Manager) available — the bootstrap installs RHODS,
  NVIDIA GPU Operator, and NFD Operator via OLM subscriptions.
- `oc` CLI authenticated with cluster-admin privileges.

## Script flow

```
test-lseval-periodic.sh (RHOAI_PROVISION=true)
│
├─ 1. Create NFD + NVIDIA namespaces
│     manifests/namespaces/{nfd,nvidia-operator}.yaml
│
├─ 2. scripts/bootstrap.sh
│     Install operator subscriptions (RHODS, GPU Operator, NFD),
│     wait for CSVs to reach Succeeded, create DataScienceCluster
│
├─ 3. scripts/gpu-setup.sh
│     Apply NFD instance + ClusterPolicy, patch tolerations,
│     wait for GPU operator pods healthy + GPU capacity on nodes
│
├─ 4. Create vLLM namespace, secrets, and chat-template ConfigMap
│
├─ 5. scripts/fetch-vllm-image.sh
│     Extract vLLM CUDA image from RHOAI ServingRuntime template
│     (falls back to a pinned registry.redhat.io digest)
│
├─ 6. scripts/deploy-vllm.sh
│     Wait for KServe CRDs + controller + webhook, re-verify GPU,
│     apply ServingRuntime + InferenceService manifests
│
├─ 7. scripts/get-vllm-pod-info.sh
│     Wait for the InferenceService pod to reach Running,
│     discover the KSVC_URL (Knative or RawDeployment), write pod.env
│
└─ 8. Run LSEval suite against the vLLM endpoint
```

## Runtime expectations

The full provisioning flow takes roughly **30–50 minutes** on a warm cluster,
dominated by:

- Operator CSV installs and reconciliation (~5–10 min)
- GPU operator pod image pulls and NVIDIA driver loading (~10–20 min)
- Llama 3.1 8B model download and vLLM startup (~10–15 min)

On a cold cluster (first GPU workload, no image cache), add another 10–15 min
for image pulls.

## Directory layout

```
tests/rhoai/
├── manifests/
│   ├── gpu/            # NFD instance, NVIDIA ClusterPolicy
│   ├── namespaces/     # NFD and NVIDIA operator namespaces
│   ├── operators/      # OLM subscriptions, OperatorGroups, DataScienceCluster
│   └── vllm/           # ServingRuntime and InferenceService for vLLM
└── scripts/
    ├── bootstrap.sh        # Install and wait for operators
    ├── gpu-setup.sh        # NFD + GPU capacity setup
    ├── fetch-vllm-image.sh # Resolve vLLM container image
    ├── deploy-vllm.sh      # Deploy vLLM via KServe
    └── get-vllm-pod-info.sh# Discover endpoint URL, write pod.env
```
