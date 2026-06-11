#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_NS="netobserv-eval-dns-latency"
export TARGET_NS
export REQUIRED_NETOBSERV_FEATURES="DNSTracking"

# shellcheck source=../common/check_prereqs.sh
source "${SCRIPT_DIR}/../common/check_prereqs.sh"
# shellcheck source=../common/wait_for.sh
source "${SCRIPT_DIR}/../common/wait_for.sh"

check_netobserv_prereqs
deploy_netobserv_fixture "${SCRIPT_DIR}/fixtures" "${TARGET_NS}"

wait_for_rollout "${TARGET_NS}" "dns-latency-prober" "180s"
echo "Waiting for DNS prober traffic (NetObserv export may take a few minutes)…"
wait_for_log_pattern "${TARGET_NS}" "app=dns-latency-prober" "kubernetes\\.default|Name:|Address" 30 3
wait_for_netobserv_warmup
echo "Scenario dns_latency ready (TARGET_NS=${TARGET_NS})"
