#!/usr/bin/env bash
set -euo pipefail

FIXTURE_DIR="$(cd "$(dirname "$0")/fixtures" && pwd)"
NS="service-mesh"

oc apply -f "$FIXTURE_DIR/manifest.yaml"

# Wait for backend to be ready
oc wait --for=condition=available deployment/backend -n "$NS" --timeout=60s

# Wait for the connection timeout error in frontend logs
ATTEMPT=0
until [ "$ATTEMPT" -ge 30 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  if oc logs -l app=frontend -n "$NS" --tail=20 2>/dev/null \
     | grep -q "ERROR: Connection timeout to backend-service!"; then
    echo "Timeout error detected (attempt $ATTEMPT)"
    exit 0
  fi
  echo "attempt $ATTEMPT/30 — waiting 3s…"
  sleep 3
done

echo "Connection timeout error not found within 90s"
oc get pods -n "$NS"
oc logs -l app=frontend -n "$NS" --tail=10 2>/dev/null || true
exit 1
