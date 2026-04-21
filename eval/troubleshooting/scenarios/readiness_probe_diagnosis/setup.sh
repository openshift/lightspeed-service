#!/usr/bin/env bash
set -euo pipefail

FIXTURE_DIR="$(cd "$(dirname "$0")/fixtures" && pwd)"
NS="discovery-hub"
POD_NAME="catalog-index-service"

oc apply -f "$FIXTURE_DIR/manifest.yaml"

# Poll for a Readiness probe failed event on the pod
ATTEMPT=0
until [ "$ATTEMPT" -ge 60 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  EVENTS=$(oc get events -n "$NS" \
    --field-selector "involvedObject.name=$POD_NAME,reason=Unhealthy" \
    -o jsonpath='{.items[*].message}' 2>/dev/null || true)
  if echo "$EVENTS" | grep -q "Readiness probe failed"; then
    echo "Readiness probe failure event found (attempt $ATTEMPT)"
    exit 0
  fi
  sleep 1
done

echo "Readiness probe failure not detected within 60s"
oc describe pod "$POD_NAME" -n "$NS"
oc get events -n "$NS" --sort-by='.lastTimestamp' | tail -15
exit 1
