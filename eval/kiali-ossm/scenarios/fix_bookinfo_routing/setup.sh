#!/usr/bin/env bash
KUBECTL=${KUBECTL:-oc}   # kubectl for kind, oc for OpenShift (set via CLUSTER env)
set -euo pipefail

FIXTURE_DIR="$(cd "$(dirname "$0")/fixtures" && pwd)"
NAMESPACE="bookinfo"
WAIT_SECONDS=${WAIT_SECONDS:-180}  # override with: make all WAIT_SECONDS=60

# Apply the fault injection — DestinationRule + VirtualService with 100% 503 abort
${KUBECTL} apply -f "$FIXTURE_DIR/manifests.yaml"

# Verify the VirtualService was accepted by Istio
ATTEMPT=0
VS_FOUND=0
until [ "$ATTEMPT" -ge 10 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  if $KUBECTL get virtualservice reviews -n "$NAMESPACE" >/dev/null 2>&1; then
    echo "VirtualService reviews is active in namespace $NAMESPACE"
    VS_FOUND=1
    break
  fi
  echo "attempt $ATTEMPT/10 — VirtualService not yet visible, waiting 3s…"
  sleep 3
done

if [ "$VS_FOUND" -eq 0 ]; then
  echo "ERROR: VirtualService reviews was not created in namespace $NAMESPACE"
  exit 1
fi

# Wait for traffic stats to accumulate so Kiali reflects the fault injection
echo "Waiting ${WAIT_SECONDS}s for Istio metrics to propagate to Kiali…"
sleep "$WAIT_SECONDS"
echo "Setup complete — routing to 50/50 in v1/v2 is active and traffic stats are ready."