#!/bin/bash
# CI job: run the LSEval periodic suite only — full 797-question QnA dataset (eval/eval_data.yaml).
# Six providers run sequentially; judge model is gpt-5-mini (from eval YAML).
# Troubleshooting evals are not run from this entrypoint (run them separately if needed).
#
# When RHOAI_PROVISION=true, the script provisions RHOAI operators, GPU infra,
# and a vLLM model serving endpoint before running OLS + evals against the
# self-hosted Llama-3.1-8B-Instruct model.
#
# After the suite completes, scores are appended to eval/score_history.csv and
# weekly trend plots are written to ARTIFACT_DIR.
#
# Input environment variables:
#   OPENAI_PROVIDER_KEY_PATH       - path to the OpenAI key (judge and OLS)
#   WATSONX_PROVIDER_KEY_PATH      - path to the WatsonX key
#   AZUREOPENAI_PROVIDER_KEY_PATH  - path to the Azure OpenAI key
#   VERTEX_PROVIDER_KEY_PATH       - path to the Vertex credentials
#   BEDROCK_AWS_ACCESS_KEY_ID      - AWS access key ID for Bedrock
#   BEDROCK_AWS_SECRET_ACCESS_KEY  - AWS secret access key for Bedrock
#   OLS_IMAGE                      - pullspec for the OLS image
#
# Additional env vars when RHOAI_PROVISION=true:
#   HUGGING_FACE_HUB_TOKEN    - download Llama 3.1 8B from HuggingFace
#   VLLM_API_KEY              - API key for the vLLM endpoint

set -eou pipefail

make install-deps && make install-deps-test
uv sync --extra evaluation --extra lseval

DIR="${BASH_SOURCE%/*}"
if [[ ! -d "$DIR" ]]; then DIR="$PWD"; fi
. "$DIR/utils.sh"

# Install operator-sdk
export ARCH=$(case $(uname -m) in x86_64) echo -n amd64 ;; aarch64) echo -n arm64 ;; *) echo -n $(uname -m) ;; esac)
export OS=$(uname | awk '{print tolower($0)}')
export OPERATOR_SDK_DL_URL=https://github.com/operator-framework/operator-sdk/releases/download/v1.36.1
curl -LO ${OPERATOR_SDK_DL_URL}/operator-sdk_${OS}_${ARCH}
mkdir -p $HOME/.local/bin
chmod +x operator-sdk_${OS}_${ARCH} && mv operator-sdk_${OS}_${ARCH} $HOME/.local/bin/operator-sdk
export PATH=$HOME/.local/bin:$PATH
operator-sdk version

# Export OpenAI key so the judge LLM can authenticate
export OPENAI_API_KEY=$(cat "$OPENAI_PROVIDER_KEY_PATH")

# ── RHOAI provisioning (conditional) ──────────────────────────────────
RHOAI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/tests/rhoai"

if [[ "${RHOAI_PROVISION:-false}" == "true" ]]; then
  echo "===== RHOAI provisioning enabled ====="

  # Validate required env vars
  : "${HUGGING_FACE_HUB_TOKEN:?HUGGING_FACE_HUB_TOKEN must be set when RHOAI_PROVISION=true}"
  : "${VLLM_API_KEY:?VLLM_API_KEY must be set when RHOAI_PROVISION=true}"

  RHOAI_NAMESPACE="e2e-rhoai-dsc"

  # 1. Create NFD and NVIDIA namespaces (needed by operator subscriptions)
  echo "--> Creating NFD and NVIDIA namespaces..."
  oc apply -f "$RHOAI_DIR/manifests/namespaces/nfd.yaml"
  oc apply -f "$RHOAI_DIR/manifests/namespaces/nvidia-operator.yaml"

  # 2. Bootstrap operators (install 3 operators, wait for CSVs, create DSC)
  echo "--> Bootstrapping RHOAI operators..."
  "$RHOAI_DIR/scripts/bootstrap.sh" "$RHOAI_DIR"

  # 3. GPU setup (NFD instance, ClusterPolicy, wait for GPU capacity)
  echo "--> Setting up GPU..."
  "$RHOAI_DIR/scripts/gpu-setup.sh" "$RHOAI_DIR"

  # 4. Create vLLM namespace and secrets
  echo "--> Creating vLLM namespace and secrets..."
  oc get ns "$RHOAI_NAMESPACE" >/dev/null 2>&1 || oc create namespace "$RHOAI_NAMESPACE"

  oc create secret generic hf-token-secret \
    --from-literal=token="$HUGGING_FACE_HUB_TOKEN" \
    -n "$RHOAI_NAMESPACE" --dry-run=client -o yaml | oc apply -f -

  oc create secret generic vllm-api-key-secret \
    --from-literal=key="$VLLM_API_KEY" \
    -n "$RHOAI_NAMESPACE" --dry-run=client -o yaml | oc apply -f -

  # 5. Create vLLM chat template ConfigMap
  echo "--> Creating vLLM chat template ConfigMap..."
  curl -sL -o /tmp/tool_chat_template_llama3.1_json.jinja \
    https://raw.githubusercontent.com/vllm-project/vllm/main/examples/tool_chat_template_llama3.1_json.jinja \
    || { echo "Failed to download jinja template"; exit 1; }

  oc create configmap vllm-chat-template -n "$RHOAI_NAMESPACE" \
    --from-file=tool_chat_template_llama3.1_json.jinja=/tmp/tool_chat_template_llama3.1_json.jinja \
    --dry-run=client -o yaml | oc apply -n "$RHOAI_NAMESPACE" -f -

  # 6. Fetch vLLM image from RHOAI template
  echo "--> Fetching vLLM image..."
  source "$RHOAI_DIR/scripts/fetch-vllm-image.sh"

  # 7. Deploy vLLM (ServingRuntime + InferenceService)
  echo "--> Deploying vLLM..."
  "$RHOAI_DIR/scripts/deploy-vllm.sh" "$RHOAI_DIR"

  # 8. Get vLLM pod info and KSVC_URL
  echo "--> Getting vLLM pod info..."
  "$RHOAI_DIR/scripts/get-vllm-pod-info.sh"
  source pod.env
  export KSVC_URL
  echo "vLLM endpoint: $KSVC_URL"

  # 9. Write VLLM_API_KEY to temp file for PROVIDER_KEY_PATH
  RHOAI_KEY_FILE=$(mktemp)
  echo -n "$VLLM_API_KEY" > "$RHOAI_KEY_FILE"
  export RHOAI_PROVIDER_KEY_PATH="$RHOAI_KEY_FILE"

  echo "===== RHOAI provisioning complete ====="
fi
# ── End RHOAI provisioning ────────────────────────────────────────────

function run_suites() {
  local rc=0

  set +e

  if [[ "${RHOAI_PROVISION:-false}" == "true" ]]; then
    # Deploy OLS with RHOAI vLLM (self-hosted Llama-3.1-8B-Instruct)
    SUITE_ID="lseval_periodic_rhoai" run_suite \
      "lseval_periodic_rhoai" "lseval" "rhoai_vllm" "$RHOAI_PROVIDER_KEY_PATH" \
      "meta-llama/Llama-3.1-8B-Instruct" "$OLS_IMAGE" "lseval"
    (( rc = rc || $? ))
  else
    # Keep the same provider order and models as the current presubmit matrix.
    SUITE_ID="lseval_periodic_openai" run_suite \
      "lseval_periodic_openai" "lseval" "openai" "$OPENAI_PROVIDER_KEY_PATH" "gpt-6-luna" "$OLS_IMAGE" "lseval" || rc=1
    SUITE_ID="lseval_periodic_watsonx" run_suite \
      "lseval_periodic_watsonx" "lseval" "watsonx" "$WATSONX_PROVIDER_KEY_PATH" "ibm/granite-4-h-small" "$OLS_IMAGE" "lseval" || rc=1
    SUITE_ID="lseval_periodic_azure_openai" run_suite \
      "lseval_periodic_azure_openai" "lseval" "azure_openai" "$AZUREOPENAI_PROVIDER_KEY_PATH" "gpt-6-luna" "$OLS_IMAGE" "lseval" || rc=1
    SUITE_ID="lseval_periodic_vertex_gemini" run_suite \
      "lseval_periodic_vertex_gemini" "lseval" "vertex_gemini" "$VERTEX_PROVIDER_KEY_PATH" "gemini-3.1-flash-lite" "$OLS_IMAGE" "lseval" || rc=1
    SUITE_ID="lseval_periodic_vertex_claude" run_suite \
      "lseval_periodic_vertex_claude" "lseval" "vertex_claude" "$VERTEX_PROVIDER_KEY_PATH" "claude-opus-4-6" "$OLS_IMAGE" "lseval" || rc=1
    SUITE_ID="lseval_periodic_bedrock_openai" run_suite \
      "lseval_periodic_bedrock_openai" "lseval" "bedrock_openai" "iam" "openai.gpt-6-luna" "$OLS_IMAGE" "lseval" || rc=1
  fi

  set -e

  cleanup_ols_operator
  return $rc
}

# lightspeed-eval writes evaluation_<timestamp>_summary.json (see eval/README.md).
_newest_eval_summary() {
  local dir="$1" f best="" best_t=-1 t
  [[ -d "$dir" ]] || return 0
  shopt -s nullglob
  for f in "$dir"/evaluation_*_summary.json; do
    [[ -f "$f" ]] || continue
    t=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    if (( t > best_t )); then
      best_t=$t
      best=$f
    fi
  done
  printf '%s' "$best"
}

function record_trends() {
  # Append periodic LSEval summary to score history and refresh trend plots.
  mkdir -p eval

  local providers=(openai watsonx azure_openai vertex_gemini vertex_claude bedrock_openai)
  if [[ "${RHOAI_PROVISION:-false}" == "true" ]]; then
    providers=(rhoai_vllm)
  fi

  local provider periodic suite_id
  for provider in "${providers[@]}"; do
    periodic="$(_newest_eval_summary "${ARTIFACT_DIR}/lseval/${provider}")"
    if [[ -z "$periodic" ]]; then
      echo "WARNING: no periodic summary under ${ARTIFACT_DIR}/lseval/${provider}, skipping trend update"
      continue
    fi

    suite_id="lseval_periodic_${provider}"
    if [[ "$provider" == "rhoai_vllm" ]]; then
      suite_id="lseval_periodic_rhoai"
    fi
    uv run --extra evaluation python eval/scripts/update_eval_trends.py \
      --history-csv eval/score_history.csv \
      --output-dir "${ARTIFACT_DIR}" \
      --suite "$suite_id" \
      --summary-json "$periodic" || true
  done
  return 0
}

function finish() {
  [[ -n "${RHOAI_KEY_FILE:-}" ]] && rm -f "$RHOAI_KEY_FILE"
  record_trends
  if [ "${LOCAL_MODE:-0}" -eq 1 ]; then
    rm -rf "$ARTIFACT_DIR"
  fi
}
trap finish EXIT

# ARTIFACT_DIR is set automatically in Prow; fall back to a temp dir locally
if [ -z "${ARTIFACT_DIR:-}" ]; then
  export ARTIFACT_DIR=$(mktemp -d)
  readonly LOCAL_MODE=1
fi

run_suites
