#!/usr/bin/env bash
set -euo pipefail

FIXTURE_DIR="$(cd "$(dirname "$0")/fixtures" && pwd)"
NS="platform-core"

# Deploy api-gateway (backend + netpol) and wait for it
oc apply -f "$FIXTURE_DIR/api-gateway.yaml"
oc wait --for=condition=available deployment/api-gateway -n "$NS" --timeout=60s

# Now deploy the web-portal (frontend) that will be blocked by the NetworkPolicy
oc apply -f "$FIXTURE_DIR/web-portal.yaml"

# Wait for the connection-timeout log line from the web-portal
ATTEMPT=0
until [ "$ATTEMPT" -ge 30 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  if oc logs -l app=web-portal -n "$NS" --tail=20 2>/dev/null \
     | grep -q "ERROR: Connection timeout to api-gateway-svc!"; then
    echo "Timeout error detected (attempt $ATTEMPT)"
    exit 0
  fi
  echo "attempt $ATTEMPT/30 — waiting 2s…"
  sleep 2
done

echo "Connection timeout error not found within 60s"
oc get pods -n "$NS"
oc logs -l app=web-portal -n "$NS" --tail=10 2>/dev/null || true
exit 1
