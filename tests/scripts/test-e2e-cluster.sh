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
  # If changes are done in this file, please make sure they reflect in test-e2e-cluster-periodics.sh and test-evaluation.sh

  # runsuite arguments:
  # suiteid test_tags provider provider_keypath model ols_image os_config_suffix
  # empty test_tags means run all tests
  run_suite "azure_openai" "not certificates and not (tool_calling and not smoketest and not rag) and not byok1 and not byok2 and not quota_limits and not data_export" "azure_openai" "$AZUREOPENAI_PROVIDER_KEY_PATH" "gpt-4o-mini" "$OLS_IMAGE" "default"
  (( rc = rc || $? ))

  run_suite "openai" "not azure_entra_id and not certificates and not (tool_calling and not smoketest and not rag) and not byok1 and not byok2 and not quota_limits and not data_export" "openai" "$OPENAI_PROVIDER_KEY_PATH" "gpt-4o-mini" "$OLS_IMAGE" "default"
  (( rc = rc || $? ))

  run_suite "watsonx" "not azure_entra_id and not certificates and not (tool_calling and not smoketest and not rag) and not byok1 and not byok2 and not quota_limits and not data_export" "watsonx" "$WATSONX_PROVIDER_KEY_PATH" "ibm/granite-4-h-small" "$OLS_IMAGE" "default"
  (( rc = rc || $? ))

  # smoke tests for RHOAI VLLM-compatible provider
  run_suite "rhoai_vllm" "smoketest" "rhoai_vllm" "$OPENAI_PROVIDER_KEY_PATH" "gpt-3.5-turbo" "$OLS_IMAGE" "default"
  (( rc = rc || $? ))

  # smoke tests for RHELAI VLLM-compatible provider
  run_suite "rhelai_vllm" "smoketest" "rhelai_vllm" "$OPENAI_PROVIDER_KEY_PATH" "gpt-3.5-turbo" "$OLS_IMAGE" "default"
  (( rc = rc || $? ))

  # TODO: Reduce execution time. Sequential execution will take more time. Parallel execution will have cluster claim issue.
  # Run tool calling - Enable tool_calling
  run_suite "azure_openai_tool_calling" "tool_calling" "azure_openai" "$AZUREOPENAI_PROVIDER_KEY_PATH" "gpt-4o-mini" "$OLS_IMAGE" "tool_calling"
  (( rc = rc || $? ))
  run_suite "openai_tool_calling" "tool_calling" "openai" "$OPENAI_PROVIDER_KEY_PATH" "gpt-4o-mini" "$OLS_IMAGE" "tool_calling"
  (( rc = rc || $? ))
  run_suite "watsonx_tool_calling" "tool_calling" "watsonx" "$WATSONX_PROVIDER_KEY_PATH" "ibm/granite-4-h-small" "$OLS_IMAGE" "tool_calling"
  (( rc = rc || $? ))

  # BYOK Test cases
  run_suite "watsonx_byok1" "byok1" "watsonx" "$WATSONX_PROVIDER_KEY_PATH" "ibm/granite-4-h-small" "$OLS_IMAGE" "byok1"
  (( rc = rc || $? ))
  run_suite "watsonx_byok2" "byok2" "watsonx" "$WATSONX_PROVIDER_KEY_PATH" "ibm/granite-4-h-small" "$OLS_IMAGE" "byok2"
  (( rc = rc || $? ))

  # quota limits tests, independent of provider therefore only testing one
  run_suite "quota_limits" "quota_limits" "openai" "$OPENAI_PROVIDER_KEY_PATH" "gpt-4o-mini" "$OLS_IMAGE" "quota"
  (( rc = rc || $? ))

  # exporter test
  run_suite "data_export" "data_export" "openai" "$OPENAI_PROVIDER_KEY_PATH" "gpt-4o-mini" "$OLS_IMAGE" "data_export"
  (( rc = rc || $? ))

  cleanup_ols_operator 

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
