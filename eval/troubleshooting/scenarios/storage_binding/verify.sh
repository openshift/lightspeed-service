#!/usr/bin/env bash
set -euo pipefail

NS="cache-tier"
PVC="memcached-data-pvc"

# Check for ProvisioningFailed via describe output (different approach from original)
oc describe pvc "$PVC" -n "$NS" 2>/dev/null \
  | grep -q "ProvisioningFailed" \
  || { echo "FAIL: ProvisioningFailed not found for $PVC"; exit 1; }

echo "PASS: ProvisioningFailed event detected for $PVC"
