#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../common/check_prereqs.sh"
cleanup_netobserv_fixture "netobserv-eval-dns-nxdomain"
