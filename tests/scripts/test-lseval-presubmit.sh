#!/bin/bash
# CI job: run the LSEval presubmit suite — short 10-question QnA dataset (eval/eval_data_short.yaml).
# Runs all 6 non-RHOAI providers sequentially. Each provider deploys OLS with the
# appropriate OLSConfig CRD, runs the 10-question eval, and tears down.
# No trend recording for presubmit smoke tests. The daily entrypoint reuses
# this setup and the shared matrix but publishes separate daily trends.
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
. "$DIR/lseval-short-common.sh"

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

function finish() {
  local rc=$?
  if [[ "${LSEVAL_RUN_KIND:-presubmit}" == "daily" ]]; then
    # Preserve the evaluation failure while still publishing any successful
    # provider summaries and their trends. History failures fail the job too.
    uv run --extra evaluation python -m eval.scripts.build_daily_eval_trends \
      --artifact-dir "$ARTIFACT_DIR" || rc=1
  elif [ "${LOCAL_MODE:-0}" -eq 1 ]; then
    rm -rf "$ARTIFACT_DIR"
  fi
  exit "$rc"
}
trap finish EXIT

# ARTIFACT_DIR is set automatically in Prow; fall back to a temp dir locally
if [ -z "${ARTIFACT_DIR:-}" ]; then
  export ARTIFACT_DIR=$(mktemp -d)
  readonly LOCAL_MODE=1
fi

run_short_lseval_suites "${LSEVAL_RUN_KIND:-presubmit}"
