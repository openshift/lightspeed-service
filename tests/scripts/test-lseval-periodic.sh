#!/bin/bash
# Periodic CI job: run LSEval against OLS using WatsonX Granite + OpenAI GPT judge.
#
# Input environment variables:
#   WATSONX_PROVIDER_KEY_PATH - path to file containing the WatsonX API key (for OLS)
#   OPENAI_PROVIDER_KEY_PATH  - path to file containing the OpenAI API key (for judge LLM)
#   OLS_IMAGE                 - pullspec for the OLS container image to deploy
#
# Script flow:
#   1. Install OLS dependencies
#   2. Install operator-sdk
#   3. Deploy OLS on the cluster (watsonx_lseval config via run_suite)
#   4. Run the LSEval periodic pytest test (make test-lseval-periodic)
#   5. Collect artefacts and clean up

set -eou pipefail

make install-deps && make install-deps-test

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
  # Deploy OLS with WatsonX Granite and run LSEval evaluation.
  # run_suite arguments: suiteid test_tags provider provider_keypath model ols_image os_config_suffix
  # OLS_CONFIG_SUFFIX="lseval" → ols_installer builds: olsconfig.crd.watsonx_lseval.yaml
  run_suite "lseval_periodic" "lseval" "watsonx" "$WATSONX_PROVIDER_KEY_PATH" "ibm/granite-4-h-small" "$OLS_IMAGE" "lseval"
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
