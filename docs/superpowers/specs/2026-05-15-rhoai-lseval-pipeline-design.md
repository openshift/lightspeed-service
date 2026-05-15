# RHOAI Self-Provisioned vLLM for ls-evals Pipeline

**Date:** 2026-05-15
**Status:** Draft
**Scope:** lightspeed-service repository

## Goal

Run ls-evals (the lightspeed evaluation framework) against a self-provisioned
RHOAI/vLLM model serving endpoint on an OpenShift cluster with GPU, as an
alternative to relying on external API providers like OpenAI. This enables
evaluating OLS performance with a self-hosted Llama 3.1 8B Instruct model
served via RHOAI's KServe/vLLM stack.

## Context

### What exists today

- `rhoai_vllm` provider type is already supported in OLS.
- `olsconfig.crd.rhoai_vllm.yaml` CRD exists for deploying OLS with that
  provider (used for smoke tests against an external endpoint).
- `system_rhoai_vllm_lseval.yaml` eval config exists, targeting `rhoai_vllm`
  provider with `gpt-5-mini` as judge LLM.
- `test-lseval-periodic.sh` handles OLS operator install + eval runs for the
  OpenAI provider.
- `ols_installer.py` handles deploying OLS via the lightspeed-operator.
- The lightspeed-stack repo (`tests/e2e-prow/rhoai/`) has proven,
  production-tested scripts for provisioning RHOAI operators, GPU
  infrastructure, and vLLM model serving on OpenShift.

### What's missing

The infrastructure to stand up RHOAI + vLLM with a GPU on the cluster from
within the lightspeed-service repo. The lightspeed-stack scripts handle
operators, GPU setup, and model deployment, but nothing in lightspeed-service
orchestrates that today.

## Design Decisions

1. **Approach: Pre-hook in existing script** -- Add a conditional RHOAI
   provisioning block at the top of `test-lseval-periodic.sh`. When
   `RHOAI_PROVISION=true`, the script provisions the GPU infra and vLLM before
   continuing into the existing OLS install + eval flow. When not set, behavior
   is unchanged.

2. **Code sharing: Copy and adapt** -- Copy the infra scripts and manifests
   from lightspeed-stack into lightspeed-service. Accept duplication for
   independence. These scripts are generic RHOAI/GPU/vLLM setup with no
   lightspeed-stack-specific logic.

3. **OLSConfig CR: Dedicated file for evals** -- Create a new
   `olsconfig.crd.rhoai_vllm_lseval.yaml` with the vLLM URL templated in
   (`${KSVC_URL}`), rather than patching the existing smoke test CR at runtime.

4. **Model: meta-llama/Llama-3.1-8B-Instruct** -- Same model used in the
   lightspeed-stack pipeline. Requires a HuggingFace token and ~16Gi GPU
   memory.

5. **Eval dataset: Full 797-question set** -- Uses `eval_data.yaml`, same as
   the existing OpenAI periodic, for comparable results across providers.

6. **Judge LLM: OpenAI gpt-5-mini** -- Unchanged from existing eval config.
   Requires `OPENAI_API_KEY` in the Prow environment.

## Architecture

### Flow

```
test-lseval-periodic.sh
|
+-- 1. Install deps (make install-deps, uv sync)
+-- 2. Install operator-sdk
+-- 3. Export OPENAI_API_KEY (for judge LLM)
|
+-- 4. RHOAI_PROVISION=true?
|   +-- 4a. Create NFD + NVIDIA namespaces
|   +-- 4b. bootstrap.sh (install 5 operators, wait for CSVs, create DSC)
|   +-- 4c. gpu-setup.sh (NFD instance, ClusterPolicy, wait for GPU capacity)
|   +-- 4d. Create vLLM namespace + secrets (hf-token, vllm-api-key)
|   +-- 4e. Create vLLM chat template ConfigMap
|   +-- 4f. fetch-vllm-image.sh (get vLLM image from RHOAI template)
|   +-- 4g. deploy-vllm.sh (ServingRuntime + InferenceService)
|   +-- 4h. get-vllm-pod-info.sh -> source pod.env -> KSVC_URL
|   +-- 4i. Write VLLM_API_KEY to temp file as PROVIDER_KEY_PATH
|   +-- 4j. Override: PROVIDER=rhoai_vllm, MODEL=Llama-3.1-8B, SUFFIX=lseval
|
+-- 5. run_suites()
|   +-- run_suite with provider/model/suffix (either RHOAI or OpenAI)
|       +-- ols_installer.py installs OLS operator
|       +-- envsubst olsconfig.crd.rhoai_vllm_lseval.yaml (injects KSVC_URL)
|       +-- Creates OLSConfig CR pointing at self-hosted vLLM
|       +-- Waits for OLS pod ready
|       +-- Runs pytest (797 questions, gpt-5-mini judge)
|
+-- 6. cleanup_ols_operator
+-- 7. record_trends (append scores to history CSV)
+-- 8. Exit
```

### Provider chain

```
ls-evals (pytest) -> OLS (via operator, exposed as route) -> vLLM (in-cluster KServe) -> GPU
                                                                |
                                                    Judge LLM: OpenAI gpt-5-mini (external)
```

## File Layout

### New files (copied from lightspeed-stack)

```
tests/e2e-prow/rhoai/
+-- manifests/
|   +-- namespaces/
|   |   +-- nfd.yaml
|   |   +-- nvidia-operator.yaml
|   +-- operators/
|   |   +-- operatorgroup.yaml
|   |   +-- operators.yaml
|   |   +-- ds-cluster.yaml
|   +-- gpu/
|   |   +-- create-nfd.yaml
|   |   +-- cluster-policy.yaml
|   +-- vllm/
|       +-- vllm-runtime-gpu.yaml
|       +-- vllm-inference-service-gpu.yaml
+-- scripts/
    +-- bootstrap.sh
    +-- gpu-setup.sh
    +-- fetch-vllm-image.sh
    +-- deploy-vllm.sh
    +-- get-vllm-pod-info.sh
```

### New file (OLSConfig CR for evals)

```
tests/config/operator_install/olsconfig.crd.rhoai_vllm_lseval.yaml
```

Contents:

```yaml
apiVersion: ols.openshift.io/v1alpha1
kind: OLSConfig
metadata:
  name: cluster
spec:
  llm:
    providers:
      - credentialsSecretRef:
          name: llmcreds
        models:
          - name: meta-llama/Llama-3.1-8B-Instruct
        name: rhoai_vllm
        type: rhoai_vllm
        url: "${KSVC_URL}/v1"
  ols:
    defaultModel: meta-llama/Llama-3.1-8B-Instruct
    defaultProvider: rhoai_vllm
    deployment:
      replicas: 1
    disableAuth: false
    logLevel: DEBUG
    userDataCollection:
      feedbackDisabled: true
      transcriptsDisabled: true
```

The `${KSVC_URL}` placeholder is substituted via `envsubst` before applying.

### Modified files

- `tests/scripts/test-lseval-periodic.sh` -- Add conditional RHOAI
  provisioning block.
- `tests/e2e/utils/ols_installer.py` -- Before calling `oc create -f` on the
  CR, run `envsubst` on the YAML file to substitute `${KSVC_URL}`. This is the
  same pattern lightspeed-stack uses for its vLLM manifests. The substitution
  should only apply when the CR filename contains `lseval` (to avoid affecting
  other CRs that don't use templates).

### Discarded from lightspeed-stack (not copied)

- Llama Stack configs, manifests, image build
- Lightspeed Stack configs, manifests
- RAG/FAISS setup
- Mock servers (JWKS, MCP)
- `pipeline-services.sh`, `pipeline.sh`, `run-tests.sh`
- Port-forwarding logic (OLS uses operator-managed routes)
- CPU vLLM variants

## Operators Installed

The RHOAI provisioning installs these 5 operators via OLM subscriptions:

| Operator | Channel | Source | Namespace |
|----------|---------|--------|-----------|
| Service Mesh | stable | redhat-operators | openshift-operators |
| Serverless | stable | redhat-operators | openshift-operators |
| RHODS (RHOAI) | stable | redhat-operators | openshift-operators |
| NVIDIA GPU | stable | certified-operators | nvidia-gpu-operator |
| Node Feature Discovery | stable | redhat-operators | openshift-nfd |

A minimal `DataScienceCluster` is created with only KServe managed (workbenches,
dashboard, pipelines all removed).

## Prow Job Environment Variables

### Required when RHOAI_PROVISION=true

| Variable | Purpose |
|----------|---------|
| `RHOAI_PROVISION` | Set to `"true"` to enable RHOAI infra provisioning |
| `HUGGING_FACE_HUB_TOKEN` | Download Llama 3.1 8B from HuggingFace |
| `VLLM_API_KEY` | API key for the vLLM endpoint (arbitrary value) |
| `OPENAI_API_KEY` | Judge model authentication (gpt-5-mini) |
| `OLS_IMAGE` | OLS container image pullspec |

### Not needed (lightspeed-stack-specific, discarded)

| Variable | Reason discarded |
|----------|------------------|
| `QUAY_ROBOT_NAME` / `QUAY_ROBOT_PASSWORD` | For pulling llama-stack images |
| `FAISS_VECTOR_STORE_ID` | RAG-specific |
| `E2E_LLAMA_HOSTNAME` | Llama Stack-specific |

## Cluster Prerequisites

- OpenShift 4.x with cluster-admin access
- GPU nodes (AWS: g4dn, g5, p3, or p4 instance types) with at least 1 NVIDIA
  GPU
- Access to `redhat-operators` and `certified-operators` catalogs in OperatorHub

## Testing Strategy

No new unit or integration tests are needed -- this is infrastructure
orchestration (shell scripts + YAML manifests). Validation happens through:

1. The pipeline itself succeeding end-to-end in the Prow job.
2. The eval results being comparable to the OpenAI-provider baseline (tracked
   via `score_history.csv` and trend plots).
3. If the RHOAI provisioning fails, the pipeline fails fast with clear error
   messages (the copied scripts already have extensive debug output and
   timeouts).

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| GPU node unavailability in Prow cluster | The gpu-setup.sh script has extensive timeouts and debug output. Prow job can be configured to require GPU node pools. |
| vLLM model loading time (can take 10+ min) | Existing scripts have 600s timeouts. The Prow job timeout should be set generously (2+ hours). |
| Operator version drift between repos | Scripts use `channel: stable` for all operators, picking up latest stable versions. If a specific version is needed, pin in operators.yaml. |
| Eval score differences vs OpenAI | Expected -- this is the point. Scores are tracked separately per provider in score_history.csv. No shared threshold enforcement initially. |
