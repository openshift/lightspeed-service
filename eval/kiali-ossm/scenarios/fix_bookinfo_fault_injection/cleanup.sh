#!/usr/bin/env bash
KUBECTL=${KUBECTL:-oc}   # kubectl for kind, oc for OpenShift (set via CLUSTER env)
set -euo pipefail

FIXTURE_DIR="$(cd "$(dirname "$0")/fixtures" && pwd)"
NAMESPACE="bookinfo"
WAIT_SECONDS=${WAIT_SECONDS:-180}  # override with: make all WAIT_SECONDS=60
KUBECTL="${KUBECTL:-oc}"

echo "Removing fault injection manifests…"
$KUBECTL delete -f "$FIXTURE_DIR/manifests.yaml" --ignore-not-found

echo "Removing any AuthorizationPolicies created by the agent during the test…"
# Scoped to the explicit names this scenario and the agent are known to create.
$KUBECTL delete authorizationpolicy allow-reviews-to-ratings  -n "$NAMESPACE" --ignore-not-found
$KUBECTL delete authorizationpolicy ratings-viewer             -n "$NAMESPACE" --ignore-not-found
$KUBECTL delete authorizationpolicy ratings-deny-all           -n "$NAMESPACE" --ignore-not-found

echo "Waiting ${WAIT_SECONDS}s for Istio metrics to stabilise after fault removal…"
sleep "$WAIT_SECONDS"
echo "Cleanup complete — fault injection and agent-created resources removed."
