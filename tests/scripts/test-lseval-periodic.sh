#!/bin/bash
# CI job: run the LSEval periodic suite only — full 797-question QnA dataset (eval/eval_data.yaml).
# OpenAI GPT-4o-mini in-cluster; judge model is gpt-5-mini (from eval YAML).
# Troubleshooting evals are not run from this entrypoint (run them separately if needed).
#
# After the suite completes, scores are appended to eval/score_history.csv and
# weekly trend plots are written to ARTIFACT_DIR.
#
# Input environment variables:
#   OPENAI_PROVIDER_KEY_PATH  - path to file containing the OpenAI API key
#   OLS_IMAGE                 - pullspec for the OLS container image to deploy

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
  # Deploy OLS with OpenAI GPT-4o-mini.
  # run_suite arguments: suiteid test_tags provider provider_keypath model ols_image ols_config_suffix
  # OLS_CONFIG_SUFFIX="lseval" → ols_installer builds: olsconfig.crd.openai_lseval.yaml
  SUITE_ID="lseval_periodic" run_suite \
    "lseval_periodic" "lseval" "openai" "$OPENAI_PROVIDER_KEY_PATH" "gpt-4o-mini" "$OLS_IMAGE" "lseval"
  (( rc = rc || $? ))

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
  local periodic
  periodic="$(_newest_eval_summary "${ARTIFACT_DIR}/lseval/openai")"
  if [[ -z "$periodic" ]]; then
    echo "WARNING: no periodic evaluation_*_summary.json under ${ARTIFACT_DIR}/lseval/openai, skipping trend update"
    return 0
  fi
  uv run --extra evaluation python eval/scripts/update_eval_trends.py \
    --history-csv eval/score_history.csv \
    --output-dir "${ARTIFACT_DIR}" \
    --suite lseval_periodic \
    --summary-json "$periodic" || true
  return 0
}

function finish() {
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
