#!/usr/bin/env bash
set -euo pipefail

NS="ingress-layer"
APP="gateway-proxy"

oc wait --for=condition=ready pod -l "app=$APP" -n "$NS" --timeout=5s 2>/dev/null \
  || { echo "FAIL: $APP pod not ready"; exit 1; }

LOGS=$(oc logs -l "app=$APP" -n "$NS" tail=10000 2>/dev/null)

echo "$LOGS" | grep -q "Configuration file change detected" \
  || { echo "FAIL: config-change sentinel missing"; exit 1; }

echo "$LOGS" | grep -q '500 GET /api/health - Connection refused' \
  || { echo "FAIL: connection-refused sentinel missing"; exit 1; }

echo "PASS: $APP has both config-change and connection-refused log lines"
