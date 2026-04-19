#!/usr/bin/env bash
set -euo pipefail

FIXTURE_DIR="$(cd "$(dirname "$0")/fixtures" && pwd)"
NS="cache-tier"
PVC="memcached-data-pvc"

oc apply -f "$FIXTURE_DIR/manifest.yaml"

# Wait for ProvisioningFailed event on the PVC
ATTEMPT=0
until [ "$ATTEMPT" -ge 60 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  STATUS=$(oc get events -n "$NS" \
    --field-selector "involvedObject.name=$PVC,involvedObject.kind=PersistentVolumeClaim" \
    -o jsonpath='{.items[*].reason}' 2>/dev/null || true)
  if echo "$STATUS" | grep -q "ProvisioningFailed"; then
    echo "ProvisioningFailed event detected (attempt $ATTEMPT)"
    exit 0
  fi
  sleep 1
done

echo "ProvisioningFailed event not detected within 60s"
oc describe pvc "$PVC" -n "$NS"
exit 1
