#!/bin/bash
# CI job: run cluster-updates eval (18 conversations, 35 evaluations) against OLS using OpenAI GPT-4o-mini + GPT-4.1-mini judge.
#
# Input environment variables:
#   OPENAI_PROVIDER_KEY_PATH  - path to file containing the OpenAI API key
#   OLS_IMAGE                 - pullspec for the OLS container image to deploy
#
# Script flow:
#   1. Install OLS dependencies
#   2. Install operator-sdk
#   3. Deploy OLS on the cluster (openai_cluster_updates config via run_suite)
#   4. Run the cluster-updates pytest test (make test-cluster-updates)
#   5. Collect artefacts and clean up

set -eou pipefail

make install-deps && make install-deps-test

DIR="${BASH_SOURCE%/*}"
if [[ ! -d "$DIR" ]]; then DIR="$PWD"; fi
. "$DIR/utils.sh"

# Install operator-sdk
ARCH=$(case $(uname -m) in x86_64) echo -n amd64 ;; aarch64) echo -n arm64 ;; *) echo -n $(uname -m) ;; esac)
export ARCH
OS=$(uname | awk '{print tolower($0)}')
export OS
export OPERATOR_SDK_DL_URL=https://github.com/operator-framework/operator-sdk/releases/download/v1.36.1
curl -LO ${OPERATOR_SDK_DL_URL}/operator-sdk_${OS}_${ARCH}
mkdir -p $HOME/.local/bin
chmod +x operator-sdk_${OS}_${ARCH} && mv operator-sdk_${OS}_${ARCH} $HOME/.local/bin/operator-sdk
export PATH=$HOME/.local/bin:$PATH
operator-sdk version

# Export OpenAI key so the judge LLM can authenticate
if [ ! -f "$OPENAI_PROVIDER_KEY_PATH" ]; then
    echo "ERROR: OPENAI_PROVIDER_KEY_PATH file not found: $OPENAI_PROVIDER_KEY_PATH" >&2
    exit 1
fi
OPENAI_API_KEY=$(cat "$OPENAI_PROVIDER_KEY_PATH")
export OPENAI_API_KEY

function run_suites() {
  local rc=0

  set +e
  echo "=== Running cluster_updates evaluation suite ==="
  echo "  Provider: openai"
  echo "  Model: gpt-4o-mini"
  echo "  Config suffix: cluster_updates"
  echo "  Artifact dir: $ARTIFACT_DIR"

  # Deploy OLS with OpenAI GPT-4o-mini and run cluster-updates evaluation (18 conversations, 35 evaluations).
  # run_suite arguments: suiteid test_tags provider provider_keypath model ols_image ols_config_suffix
  # OLS_CONFIG_SUFFIX="cluster_updates" → ols_installer builds: olsconfig.crd.openai_cluster_updates.yaml
  run_suite "cluster_updates" "cluster_updates" "openai" "$OPENAI_PROVIDER_KEY_PATH" "gpt-4o-mini" "$OLS_IMAGE" "cluster_updates"
  rc=$?

  if [ $rc -ne 0 ]; then
      echo "ERROR: cluster_updates suite failed with exit code $rc" >&2
  fi
  set -e

  echo "=== Cleaning up OLS operator ==="
  cleanup_ols_operator

  return $rc
}

function finish() {
  if [ "${LOCAL_MODE:-0}" -eq 1 ]; then
    rm -rf "$ARTIFACT_DIR"
  fi
}
trap finish EXIT

# Define LOCAL_MODE before the conditional
LOCAL_MODE=0

# ARTIFACT_DIR is set automatically in Prow; fall back to a temp dir locally
if [ -z "${ARTIFACT_DIR:-}" ]; then
  ARTIFACT_DIR=$(mktemp -d)
  LOCAL_MODE=1
fi

export ARTIFACT_DIR
export LOCAL_MODE
readonly LOCAL_MODE

run_suites
