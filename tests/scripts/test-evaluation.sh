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
  # If changes are done in this file, please make sure they reflect in test-e2e-cluster-periodics.sh and test-e2e-cluster.sh

  # runsuite arguments:
  # suiteid test_tags provider provider_keypath model ols_image os_config_suffix
  run_suite "model_eval" "" "watsonx openai azure_openai" "$WATSONX_PROVIDER_KEY_PATH $OPENAI_PROVIDER_KEY_PATH $AZUREOPENAI_PROVIDER_KEY_PATH" "ibm/granite-3-3-8b-instruct gpt-4o-mini gpt-4o-mini" "$OLS_IMAGE" "default"
  (( rc = rc || $? ))
  set -e

  cleanup_ols_operator 

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
