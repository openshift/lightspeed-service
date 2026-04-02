#!/usr/bin/env bash
set -euo pipefail

NS="warehouse-ops"
APP="order-fulfillment-daemon"

# The pod may cycle between CrashLoopBackOff and waiting states; retry a few times
ATTEMPT=0
until [ "$ATTEMPT" -ge 10 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  REASON=$(oc get pods -n "$NS" -l "app=$APP" \
    -o jsonpath='{.items[0].status.containerStatuses[0].state.waiting.reason}' 2>/dev/null || true)
  if [ "$REASON" = "CrashLoopBackOff" ]; then
    echo "PASS: $APP is in CrashLoopBackOff"
    exit 0
  fi
  sleep 2
done

echo "FAIL: $APP not in CrashLoopBackOff (got: ${REASON:-<none>})"
exit 1
