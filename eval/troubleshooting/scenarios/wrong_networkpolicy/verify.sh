#!/usr/bin/env bash
set -euo pipefail

NS="service-mesh"

# backend must be healthy
oc wait --for=condition=available deployment/backend -n "$NS" --timeout=5s \
  || { echo "FAIL: backend deployment not available"; exit 1; }

# frontend logs must contain the timeout error
oc logs -l app=frontend -n "$NS" --tail=50 2>/dev/null \
  | grep -q "ERROR: Connection timeout to backend-service!" \
  || { echo "FAIL: no connection timeout error in frontend logs"; exit 1; }

echo "PASS: backend available, frontend reports connection timeout"
