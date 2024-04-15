#!/bin/bash
set -eou pipefail

# Input env variables:
# - PROVIDER - the LLM provider to be used during the test
# - PROVIDER_KEY_PATH - path to a file containing the credentials to be used with the llm provider
# - MODEL - name of the model to use during e2e testing
# - PROVIDER_URL - url of the provider api endpoint (required for azure-openai)
# - PROVIDER_DEPLOYMENT_NAME - required for azure-openai
# - PROVIDER_PROJECT_ID - required for watsonx

# Script flow:
# 1) Generate an OLS config file from env variables
# 2) Launch an OLS server via the run Makefile target and waits for it to be ready
# 3) Invoke the test-e2e Makefile target
# 4) Terminate the OLS server

function wait_for_ols() {
  for i in {1..30}; do
    echo Checking OLS readiness, attempt "$i" of 30
    curl -sk --fail "$1/readiness"
    if [ $? -eq 0 ]; then
      return 0
    fi  
    sleep 6
  done
  return 1
}

# temp directory for config file, output logs
TMPDIR=$(mktemp -d)

# ARTIFACT_DIR is defined when running in a prow job
ARTIFACT_DIR=${ARTIFACT_DIR:-$TMPDIR}

# configure feedback storage location
export FEEDBACK_STORAGE_LOCATION="$ARTIFACT_DIR/user-data/feedback"
echo "Feedback storage location: $FEEDBACK_STORAGE_LOCATION"
export TRANSCRIPTS_STORAGE_LOCATION="$ARTIFACT_DIR/user-data/transcripts"
echo "Transcripts storage location: $TRANSCRIPTS_STORAGE_LOCATION"

export OLS_CONFIG_FILE="$ARTIFACT_DIR/olsconfig.yaml"
OLS_LOGS=$ARTIFACT_DIR/ols.log

# Generate ols config.yaml
export PROVIDER="${PROVIDER:-openai}"
export PROVIDER_KEY_PATH="${PROVIDER_KEY_PATH:-openai_api_key.txt}"
if [ ! -e "$PROVIDER_KEY_PATH" ]; then
  echo "No key found at $PROVIDER_KEY_PATH"
  exit 1
fi

export MODEL="${MODEL:-gpt-3.5-turbo-1106}"
envsubst < $(pwd)/tests/config/singleprovider.e2e.template.config.yaml > "${OLS_CONFIG_FILE}.tmp"

# If no provider url is being specified, remove the url field from the config yaml
# so we use the default provider url values.
if [ -z ${PROVIDER_URL:-} ]; then
  grep -v url: "${OLS_CONFIG_FILE}.tmp" > "$OLS_CONFIG_FILE"
  rm "${OLS_CONFIG_FILE}.tmp"
else
  mv "${OLS_CONFIG_FILE}.tmp" "${OLS_CONFIG_FILE}"
fi

echo "Installing dependencies"
make install-deps && make install-deps-test

echo Starting OLS server
make run >& "$OLS_LOGS" &
function finish() {
    echo Exit trap: killing OLS server
    kill %1
    rm -rf "$TMPDIR"
}
trap finish EXIT

set +e
wait_for_ols "localhost:8080"
if [ $? -ne 0 ]; then
  echo "Timed out waiting for OLS to start, OLS log output:"
  cat "$OLS_LOGS"
  echo "Config file:"
  cat "$OLS_CONFIG_FILE"
  exit 1
fi
set -e

echo Done waiting for OLS server start, running e2e
OLS_URL="http://localhost:8080" SUITE_ID=standalone TEST_TAGS="not cluster and not rag and not cluster_with_collector" make test-e2e
