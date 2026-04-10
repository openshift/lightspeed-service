#!/usr/bin/env bash
set -euo pipefail

NS="oom-scenario"
APP="awesome-application"

# Check that at least one pod has been OOMKilled
ATTEMPT=0
until [ "$ATTEMPT" -ge 10 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  STATUSES=$(oc get pods -n "$NS" -l "app=$APP" \
    -o jsonpath='{range .items[*]}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}' 2>/dev/null || true)
  if echo "$STATUSES" | grep -q "OOMKilled"; then
    echo "PASS: $APP has OOMKilled containers"
    exit 0
  fi
  sleep 2
done

echo "FAIL: no OOMKilled containers found for $APP"
exit 1
