#!/usr/bin/env bash
set -euo pipefail

FIXTURE_DIR="$(cd "$(dirname "$0")/fixtures" && pwd)"
NS="catalog-mgmt"
JOB="inventory-sync-validator"

oc create namespace "$NS" 2>/dev/null || true

oc create secret generic inventory-sync-logs-script \
  --from-file=generate_logs.py="$FIXTURE_DIR/generate_logs.py" \
  -n "$NS" --dry-run=client -o yaml | oc apply -f -
oc apply -f "$FIXTURE_DIR/job.yaml"

# Wait for the job pod to produce the expected log sentinels
ATTEMPT=0
until [ "$ATTEMPT" -ge 20 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  LOGS=$(oc logs -l "job-name=$JOB" -n "$NS" --tail=100 2>/dev/null || true)
  if echo "$LOGS" | grep -q "Target host: prod-db, port: 3333" \
  && echo "$LOGS" | grep -q "FATAL: Unable to connect to required database"; then
    echo "Both sentinels found (attempt $ATTEMPT)"
    exit 0
  fi
  echo "attempt $ATTEMPT/20 — waiting 3s…"
  sleep 3
done

echo "Sentinels not found within 60s"
oc logs -l "job-name=$JOB" -n "$NS" --tail=30 2>/dev/null || true
exit 1
