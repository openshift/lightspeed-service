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
| `HUGGING_FACE_HUB_TOKEN` | HuggingFace token used to download the selected model weights. Its account must have access to the selected model, including any required model-license acceptance. |
| `VLLM_API_KEY` | Arbitrary value. The flow creates the vLLM endpoint secret from it and passes the same value to OLS as the provider key. |
| `OPENAI_PROVIDER_KEY_PATH` | Path to file containing the OpenAI API key for the judge LLM. |
| `OLS_IMAGE` | OLS container image pullspec to deploy. |
| `VLLM_MODEL_PROFILE` | Optional vLLM profile. Defaults to `llama-3.1-8b`; set it to `gemma-4-31b` to serve Gemma. |

## Model profiles and consumer requirements

`tests/rhoai/scripts/model-profile.sh` is the single source of profile values.
`deploy-vllm.sh` loads the selected profile before rendering the manifests. A
consumer such as the agentic-operator product-E2E provisioning flow must load
the same profile before creating the chat-template ConfigMap and before passing
the selected `VLLM_MODEL` to its OLS configuration. Do not set individual
`VLLM_*` values: the selected profile overwrites them so the chat template,
model, parser, GPU count, tensor parallelism, and resource requests stay
consistent.

| Profile | Model and parser | GPU topology | Chat template |
|---|---|---|---|
| `llama-3.1-8b` (default) | `meta-llama/Llama-3.1-8B-Instruct`; `llama3_json` | 1 GPU; tensor parallelism 1 | vLLM Llama 3.1 JSON template |
| `gemma-4-31b` | `google/gemma-4-31B-it`; `gemma4` | 4 GPUs; tensor parallelism 4 | Gemma 4 template from Hugging Face |

Before deploying either profile, the provisioning consumer downloads the
selected `VLLM_CHAT_TEMPLATE_URL`, creates `VLLM_CHAT_TEMPLATE_CONFIGMAP`, and
stores the file using `VLLM_CHAT_TEMPLATE_KEY`. The ServingRuntime mounts that exact key as
`/mnt/chat-template/chat_template.jinja`; consequently, ConfigMap creation must
complete before `deploy-vllm.sh` applies the manifests.

For `gemma-4-31b`, use an AWS `g5.12xlarge` worker (or equivalent) with **four
NVIDIA A10G GPUs** and run tensor parallelism **4**. The profile requests four
GPUs in both the ServingRuntime and InferenceService, plus 32 CPU / 128 GiB
memory requests and 40 CPU / 160 GiB limits. Ensure the cluster has sufficient
allocatable capacity after GPU Operator overhead.

Validate Gemma on the target RHOAI release before a full LSEval run: its bundled
vLLM image must accept `--enable-auto-tool-choice` with
`--tool-call-parser gemma4`, download `google/gemma-4-31B-it`, and reach Ready.
After deployment, inspect the InferenceService and vLLM pod logs for parser or
chat-template errors, then send a tool-call request through the endpoint. This
validates the RHOAI vLLM image and parser, not just manifest rendering.

## Cluster prerequisites

- OpenShift 4.x cluster with GPU-capable nodes (e.g. AWS `g4dn`, `g5`, `p3`, `p4` instance types). The Gemma profile specifically requires a `g5.12xlarge`-class node with four A10G GPUs.
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
├─ 4. Create vLLM namespace, secrets, and selected chat-template ConfigMap
│
├─ 5. scripts/fetch-vllm-image.sh
│     Extract vLLM CUDA image from RHOAI ServingRuntime template
│     (falls back to a pinned registry.redhat.io digest)
│
├─ 6. scripts/deploy-vllm.sh
│     Wait for KServe CRDs + controller + webhook, re-verify GPU,
│     render the selected profile, then apply ServingRuntime + InferenceService manifests
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
- Llama 3.1 8B model download and vLLM startup (~10–15 min); Gemma 4 31B takes longer and requires the four-GPU profile

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
    ├── model-profile.sh    # Select model, parser, template, and resources
    ├── deploy-vllm.sh      # Render profile and deploy vLLM via KServe
    └── get-vllm-pod-info.sh# Discover endpoint URL, write pod.env
```
