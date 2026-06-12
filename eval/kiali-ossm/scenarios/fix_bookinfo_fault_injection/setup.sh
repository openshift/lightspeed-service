#!/usr/bin/env bash
KUBECTL=${KUBECTL:-oc}   # kubectl for kind, oc for OpenShift (set via CLUSTER env)
set -euo pipefail

FIXTURE_DIR="$(cd "$(dirname "$0")/fixtures" && pwd)"
NAMESPACE="bookinfo"
WAIT_SECONDS=${WAIT_SECONDS:-180}  # override with: make all WAIT_SECONDS=60
KUBECTL="${KUBECTL:-oc}"

# ── Clean pre-existing ratings Istio resources ────────────────────────────────
# Removes leftover resources — including any AuthorizationPolicies created by a
# previous agent run — so the agent starts with a clean slate each time.
echo "Removing existing ratings Istio resources…"
$KUBECTL delete virtualservice     ratings                  -n "$NAMESPACE" --ignore-not-found
$KUBECTL delete destinationrule    ratings                  -n "$NAMESPACE" --ignore-not-found
$KUBECTL delete peerauthentication ratings-permissive-mtls  -n "$NAMESPACE" --ignore-not-found
# Remove the specific AuthorizationPolicies this scenario and the agent are known to create.
# Scoped to an explicit list to avoid accidentally removing unrelated security resources.
$KUBECTL delete authorizationpolicy allow-reviews-to-ratings  -n "$NAMESPACE" --ignore-not-found
$KUBECTL delete authorizationpolicy ratings-viewer             -n "$NAMESPACE" --ignore-not-found
$KUBECTL delete authorizationpolicy ratings-deny-all           -n "$NAMESPACE" --ignore-not-found
sleep 5   # allow Istio to propagate the deletions

# ── Apply the fault injection manifests ───────────────────────────────────────
echo "Applying fault injection manifests…"
$KUBECTL apply -f "$FIXTURE_DIR/manifests.yaml"

# Verify the VirtualService was accepted
ATTEMPT=0
VS_FOUND=0
until [ "$ATTEMPT" -ge 10 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  if $KUBECTL get virtualservice ratings -n "$NAMESPACE" >/dev/null 2>&1; then
    echo "VirtualService ratings is active in namespace $NAMESPACE"
    VS_FOUND=1
    break
  fi
  echo "attempt $ATTEMPT/10 — VirtualService not yet visible, waiting 3s…"
  sleep 3
done

if [ "$VS_FOUND" -eq 0 ]; then
  echo "ERROR: VirtualService ratings was not created in namespace $NAMESPACE"
  exit 1
fi

echo "Waiting ${WAIT_SECONDS}s for Istio metrics to propagate to Kiali…"
sleep "$WAIT_SECONDS"
echo "Setup complete — fault injection is active."
