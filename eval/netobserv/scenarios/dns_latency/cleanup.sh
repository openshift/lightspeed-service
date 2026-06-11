#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../common/check_prereqs.sh
source "${SCRIPT_DIR}/../common/check_prereqs.sh"
cleanup_netobserv_fixture "netobserv-eval-dns-latency"
