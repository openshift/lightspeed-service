#!/bin/sh
# Input env variables:
# - PROVIDER - the LLM provider to be used during the test
# - PROVIDER_KEY_PATH - path to a file containing the credentials to be used with the llm provider
# - MODEL - name of the model to use during e2e testing

# Script flow:
# 1) Generate an OLS config file from env variables
# 2) Launch an OLS server via the run Makefile target and waits for it to be ready
# 3) Invoke the test-e2e Makefile target
# 4) Terminate the OLS server

# temp directory for config file, output logs
TMPDIR=$(mktemp -d)

# ARTIFACT_DIR is defined when running in a prow job
ARTIFACT_DIR=${ARTIFACT_DIR:-$TMPDIR}

# temp directory for index store
RAG_TMP_DIR=$(mktemp -d)
# Download index store
RAG_INDEX="https://github.com/ilan-pinto/lightspeed-rag-documents/releases/latest/download/local.zip"
export RAG_INDEX_DIR="$RAG_TMP_DIR/vector-db/ocp-product-docs"
# Configure index store
mkdir -p $RAG_INDEX_DIR \
  && wget $RAG_INDEX \
	&& unzip -j local.zip -d $RAG_INDEX_DIR \
	&& rm -f local.zip

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
envsubst < $(pwd)/tests/config/singleprovider.e2e.template.config.yaml > "$OLS_CONFIG_FILE"

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

STARTED=0
for i in {1..10}; do
  echo Checking OLS readiness, attempt "$i" of 10
  curl -s localhost:8080/readiness
  if [ $? -eq 0 ]; then
    STARTED=1
    break
  fi  
  sleep 6
done

if [ $STARTED -ne 1 ]; then
  echo "OLS failed to start, OLS log output:"
  cat "$OLS_LOGS"
  echo "Config file:"
  cat "$OLS_CONFIG_FILE"
  exit 1
fi

echo Done waiting for OLS server start, running e2e
make test-e2e
