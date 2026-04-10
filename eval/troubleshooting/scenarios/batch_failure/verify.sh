#!/usr/bin/env bash
set -euo pipefail

NS="catalog-mgmt"
JOB="inventory-sync-validator"

LOGS=$(oc logs -l "job-name=$JOB" -n "$NS" --tail=100 2>/dev/null)
[ -n "$LOGS" ] || { echo "FAIL: no logs from $JOB pods"; exit 1; }

echo "$LOGS" | grep -q "Target host: prod-db, port: 3333" \
  || { echo "FAIL: missing 'Target host' sentinel"; exit 1; }

echo "$LOGS" | grep -q "FATAL: Unable to connect to required database" \
  || { echo "FAIL: missing 'FATAL' sentinel"; exit 1; }

echo "PASS: $JOB pod has expected database-connection-failure logs"
