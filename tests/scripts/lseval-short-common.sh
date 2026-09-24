#!/bin/bash
# Shared presubmit/daily short-dataset provider matrix. Keep model choices here.
# Caller sources utils.sh and supplies ARTIFACT_DIR and OLS_IMAGE.

function run_short_lseval_suites() {
  local kind="$1" rc=0 suite
  if [[ "$kind" != "presubmit" && "$kind" != "daily" ]]; then
    echo "Unknown short LSEval run kind: $kind" >&2
    return 2
  fi

  # Run every provider even if an earlier suite fails. `run_suite` selects the
  # short-dataset Make target for both prefixes.
  set +e
  suite="lseval_${kind}_openai"
  SUITE_ID="$suite" run_suite "$suite" "lseval" "openai" "$OPENAI_PROVIDER_KEY_PATH" "gpt-5.4-mini" "$OLS_IMAGE" "lseval" || rc=1
  suite="lseval_${kind}_watsonx"
  SUITE_ID="$suite" run_suite "$suite" "lseval" "watsonx" "$WATSONX_PROVIDER_KEY_PATH" "ibm/granite-4-h-small" "$OLS_IMAGE" "lseval" || rc=1
  suite="lseval_${kind}_azure_openai"
  SUITE_ID="$suite" run_suite "$suite" "lseval" "azure_openai" "$AZUREOPENAI_PROVIDER_KEY_PATH" "gpt-5.4-mini" "$OLS_IMAGE" "lseval" || rc=1
  suite="lseval_${kind}_vertex_gemini"
  SUITE_ID="$suite" run_suite "$suite" "lseval" "vertex_gemini" "$VERTEX_PROVIDER_KEY_PATH" "gemini-3.1-flash-lite" "$OLS_IMAGE" "lseval" || rc=1
  suite="lseval_${kind}_vertex_claude"
  SUITE_ID="$suite" run_suite "$suite" "lseval" "vertex_claude" "$VERTEX_PROVIDER_KEY_PATH" "claude-opus-4-6" "$OLS_IMAGE" "lseval" || rc=1
  suite="lseval_${kind}_bedrock_deepseek"
  SUITE_ID="$suite" run_suite "$suite" "lseval" "bedrock_deepseek" "iam" "deepseek.v3.2" "$OLS_IMAGE" "lseval" || rc=1
  set -e

  cleanup_ols_operator || rc=1
  return "$rc"
}
