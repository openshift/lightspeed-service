#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_NS="netobserv-eval-drops-policy"
export TARGET_NS
export REQUIRED_NETOBSERV_FEATURES="NetworkEvents"

source "${SCRIPT_DIR}/../common/check_prereqs.sh"
source "${SCRIPT_DIR}/../common/wait_for.sh"

check_netobserv_prereqs
deploy_netobserv_fixture "${SCRIPT_DIR}/fixtures" "${TARGET_NS}"

wait_for_rollout "${TARGET_NS}" "policy-backend" "120s"
wait_for_rollout "${TARGET_NS}" "policy-frontend" "120s"
wait_for_log_pattern "${TARGET_NS}" "app=policy-frontend" "connection blocked|timed out" 40 3
echo "Scenario packet_drops_policy ready (TARGET_NS=${TARGET_NS})"
