#!/usr/bin/env bash
set -euo pipefail

NS="discovery-hub"
POD_NAME="catalog-index-service"

# Confirm the pod exists
oc get pod "$POD_NAME" -n "$NS" >/dev/null 2>&1 \
  || { echo "FAIL: pod $POD_NAME not found"; exit 1; }

# Check for Readiness probe failed via jsonpath on events
MESSAGES=$(oc get events -n "$NS" \
  --field-selector "involvedObject.name=$POD_NAME,reason=Unhealthy" \
  -o jsonpath='{.items[*].message}' 2>/dev/null)

echo "$MESSAGES" | grep -q "Readiness probe failed" \
  || { echo "FAIL: Readiness probe failure event not found for $POD_NAME"; exit 1; }

echo "PASS: $POD_NAME has readiness probe failures"
