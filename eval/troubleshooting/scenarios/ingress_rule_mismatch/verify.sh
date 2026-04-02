#!/usr/bin/env bash
set -euo pipefail

NS="platform-core"

# api-gateway must be healthy
oc wait --for=condition=available deployment/api-gateway -n "$NS" --timeout=5s \
  || { echo "FAIL: api-gateway deployment not available"; exit 1; }

# web-portal logs must contain the timeout error
oc logs -l app=web-portal -n "$NS" --tail=50 2>/dev/null \
  | grep -q "ERROR: Connection timeout to api-gateway-svc!" \
  || { echo "FAIL: no connection timeout error in web-portal logs"; exit 1; }

echo "PASS: api-gateway available, web-portal reports connection timeout"
