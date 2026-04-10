#!/usr/bin/env bash
set -euo pipefail

FIXTURE_DIR="$(cd "$(dirname "$0")/fixtures" && pwd)"
NS="ingress-layer"
APP="gateway-proxy"

oc create namespace "$NS" 2>/dev/null || true

# Inject the Python log generator as a secret, then apply the deployment
oc create secret generic gateway-proxy-log-script \
  --from-file=generate_logs.py="$FIXTURE_DIR/generate_logs.py" \
  -n "$NS" --dry-run=client -o yaml | oc apply -f -
oc apply -f "$FIXTURE_DIR/manifest.yaml"

# Wait until the pod is ready and both sentinel log lines are present
ATTEMPT=0
until [ "$ATTEMPT" -ge 40 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  if oc wait --for=condition=ready pod -l "app=$APP" -n "$NS" --timeout=2s 2>/dev/null; then
    LOGS=$(oc logs -l "app=$APP" -n "$NS" --tail=10000 2>/dev/null || true)
    if echo "$LOGS" | grep -q "Configuration file change detected" \
    && echo "$LOGS" | grep -q '500 GET /api/health - Connection refused'; then
      echo "Log sentinels found after $ATTEMPT checks"
      exit 0
    fi
  fi
  echo "check $ATTEMPT/40 — waiting 3s…"
  sleep 3
done

echo "Sentinels not found within timeout"
oc logs -l "app=$APP" -n "$NS" --tail=30 2>/dev/null || true
exit 1
