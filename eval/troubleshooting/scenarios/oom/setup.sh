#!/usr/bin/env bash
set -euo pipefail

FIXTURE_DIR="$(cd "$(dirname "$0")/fixtures" && pwd)"
NS="oom-scenario"
APP="awesome-application"

oc apply -f "$FIXTURE_DIR/manifest.yaml"

# Wait for at least one pod to hit OOMKilled
echo "Waiting for $APP to be OOMKilled…"
ATTEMPT=0
until [ "$ATTEMPT" -ge 90 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  STATUSES=$(oc get pods -n "$NS" -l "app=$APP" \
    -o jsonpath='{range .items[*]}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}' 2>/dev/null || true)
  if echo "$STATUSES" | grep -q "OOMKilled"; then
    echo "OOMKilled detected (attempt $ATTEMPT)"
    exit 0
  fi
  sleep 2
done

echo "OOMKilled not detected within timeout"
oc get pods -n "$NS" -l "app=$APP"
oc describe pods -n "$NS" -l "app=$APP" | tail -20
exit 1
