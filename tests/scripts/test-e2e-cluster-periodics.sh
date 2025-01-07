#!/bin/bash
set -eou pipefail

# Input env variables:
# - [PROVIDERNAME]_PROVIDER_KEY_PATH - path to a file containing the credentials to be used with the llm provider
# - OLS_IMAGE - pullspec for the ols image to deploy on the cluster


# Script flow:
# 1) Install OLS into a namespace on the cluster with valid config/api tokens and exposed via a route
# 2) Setup a service account w/ permission to access OLS
# 3) Wait for the ols api server to be available
# 4) Invoke the test-e2e Makefile target

make install-deps && make install-deps-test

DIR="${BASH_SOURCE%/*}"
if [[ ! -d "$DIR" ]]; then DIR="$PWD"; fi
. "$DIR/utils.sh"

# install operator-sdk 
export ARCH=$(case $(uname -m) in x86_64) echo -n amd64 ;; aarch64) echo -n arm64 ;; *) echo -n $(uname -m) ;; esac)
export OS=$(uname | awk '{print tolower($0)}')
export OPERATOR_SDK_DL_URL=https://github.com/operator-framework/operator-sdk/releases/download/v1.36.1
curl -LO ${OPERATOR_SDK_DL_URL}/operator-sdk_${OS}_${ARCH}
mkdir -p $HOME/.local/bin
chmod +x operator-sdk_${OS}_${ARCH} && mv operator-sdk_${OS}_${ARCH} $HOME/.local/bin/operator-sdk
export PATH=$HOME/.local/bin:$PATH
operator-sdk version

function run_suites() {
  local rc=0

  set +e
  # runsuite arguments:
  # suiteid test_tags provider provider_keypath model ols_image
  # empty test_tags means run all tests
  run_suite "azure_openai" "not model_evaluation and not certificates" "azure_openai" "$AZUREOPENAI_PROVIDER_KEY_PATH" "gpt-4o-mini" "$OLS_IMAGE"
  (( rc = rc || $? ))

  # BAM is currently not working, commenting for now
  # run_suite "bam" "not model_evaluation" "bam" "$BAM_PROVIDER_KEY_PATH" "ibm/granite-3-8b-instruct" "$OLS_IMAGE"
  # (( rc = rc || $? ))

  run_suite "openai" "not model_evaluation and not azure_entra_id and not certificates" "openai" "$OPENAI_PROVIDER_KEY_PATH" "gpt-4o-mini" "$OLS_IMAGE"
  (( rc = rc || $? ))

  run_suite "watsonx" " not azure_entra_id and not certificates" "watsonx" "$WATSONX_PROVIDER_KEY_PATH" "ibm/granite-3-8b-instruct" "$OLS_IMAGE"
  (( rc = rc || $? ))

  # smoke tests for RHOAI VLLM-compatible provider
  run_suite "rhoai_vllm" "smoketest" "rhoai_vllm" "$OPENAI_PROVIDER_KEY_PATH" "gpt-3.5-turbo" "$OLS_IMAGE"
  (( rc = rc || $? ))

  # smoke tests for RHELAI VLLM-compatible provider
  run_suite "rhelai_vllm" "smoketest" "rhelai_vllm" "$OPENAI_PROVIDER_KEY_PATH" "gpt-3.5-turbo" "$OLS_IMAGE"
  (( rc = rc || $? ))
  
  run_suite "openai" "certificates" "openai" "$OPENAI_PROVIDER_KEY_PATH" "gpt-4o-mini" "$OLS_IMAGE"
  (( rc = rc || $? ))

  set -e

  return $rc
}

function finish() {
    if [ "${LOCAL_MODE:-0}" -eq 1 ]; then
      # When running locally, cleanup the tmp files
      rm -rf "$ARTIFACT_DIR"
    fi
}
trap finish EXIT

# ARTIFACT_DIR is defined when running in a prow job, content
# in this location is automatically collected at the end of the test job
# If ARTIFACT_DIR is not defined, we are running locally on a developer machine
if [ -z "${ARTIFACT_DIR:-}" ]; then
    # temp directory for generated resource yamls
    export ARTIFACT_DIR=$(mktemp -d)
    # Clean up the tmpdir on exit
    readonly LOCAL_MODE=1
fi

run_suites
