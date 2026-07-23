#!/bin/bash
# CI job: run the LSEval presubmit suite — short 10-question QnA dataset (eval/eval_data_short.yaml).
# Runs all 6 non-RHOAI providers sequentially. Each provider deploys OLS with the
# appropriate OLSConfig CRD, runs the 10-question eval, and tears down.
# No trend recording — presubmit runs are smoke tests.
#
# Input environment variables:
#   OPENAI_PROVIDER_KEY_PATH       - path to file containing the OpenAI API key (judge LLM + OLS)
#   AZUREOPENAI_PROVIDER_KEY_PATH  - path to file containing the Azure OpenAI API key
#   WATSONX_PROVIDER_KEY_PATH      - path to file containing the WatsonX API key
#   VERTEX_PROVIDER_KEY_PATH       - path to file containing the Vertex AI credentials
#   BEDROCK_AWS_ACCESS_KEY_ID      - AWS access key ID for Bedrock IAM
#   BEDROCK_AWS_SECRET_ACCESS_KEY  - AWS secret access key for Bedrock IAM
#   OLS_IMAGE                      - pullspec for the OLS container image to deploy

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

function run_suites() {
  local rc=0
  set +e

  # OpenAI
  SUITE_ID="lseval_presubmit_openai" run_suite \
    "lseval_presubmit_openai" "lseval" "openai" "$OPENAI_PROVIDER_KEY_PATH" "gpt-5.4-mini" "$OLS_IMAGE" "lseval"
  (( rc = rc || $? ))

  # WatsonX
  SUITE_ID="lseval_presubmit_watsonx" run_suite \
    "lseval_presubmit_watsonx" "lseval" "watsonx" "$WATSONX_PROVIDER_KEY_PATH" "ibm/granite-4-h-small" "$OLS_IMAGE" "lseval"
  (( rc = rc || $? ))

  # Azure OpenAI
  SUITE_ID="lseval_presubmit_azure_openai" run_suite \
    "lseval_presubmit_azure_openai" "lseval" "azure_openai" "$AZUREOPENAI_PROVIDER_KEY_PATH" "gpt-5.4-mini" "$OLS_IMAGE" "lseval"
  (( rc = rc || $? ))

  # Vertex Gemini
  SUITE_ID="lseval_presubmit_vertex_gemini" run_suite \
    "lseval_presubmit_vertex_gemini" "lseval" "vertex_gemini" "$VERTEX_PROVIDER_KEY_PATH" "gemini-3.5-flash" "$OLS_IMAGE" "lseval"
  (( rc = rc || $? ))

  # Vertex Claude
  SUITE_ID="lseval_presubmit_vertex_claude" run_suite \
    "lseval_presubmit_vertex_claude" "lseval" "vertex_claude" "$VERTEX_PROVIDER_KEY_PATH" "claude-opus-4-6" "$OLS_IMAGE" "lseval"
  (( rc = rc || $? ))

  # Bedrock DeepSeek
  SUITE_ID="lseval_presubmit_bedrock_deepseek" run_suite \
    "lseval_presubmit_bedrock_deepseek" "lseval" "bedrock_deepseek" "iam" "deepseek-r1" "$OLS_IMAGE" "lseval"
  (( rc = rc || $? ))

  set -e
  cleanup_ols_operator
  return $rc
}

function finish() {
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
