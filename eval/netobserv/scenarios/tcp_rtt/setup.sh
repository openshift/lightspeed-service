#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_NS="netobserv-eval-tcp-rtt"
export TARGET_NS
export REQUIRED_NETOBSERV_FEATURES="FlowRTT"

source "${SCRIPT_DIR}/../common/check_prereqs.sh"
source "${SCRIPT_DIR}/../common/wait_for.sh"

check_netobserv_prereqs
deploy_netobserv_fixture "${SCRIPT_DIR}/fixtures" "${TARGET_NS}"

wait_for_rollout "${TARGET_NS}" "slow-http-server" "120s"
wait_for_rollout "${TARGET_NS}" "rtt-client" "120s"
wait_for_log_pattern "${TARGET_NS}" "app=rtt-client" "OK round-trip|elevated RTT" 40 3
echo "Scenario tcp_rtt ready (TARGET_NS=${TARGET_NS})"
