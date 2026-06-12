#!/usr/bin/env bash
set -euo pipefail

FIXTURE_DIR="$(cd "$(dirname "$0")/fixtures" && pwd)"
NAMESPACE="bookinfo"
WAIT_SECONDS=${WAIT_SECONDS:-180}  # override with: make all WAIT_SECONDS=60
KUBECTL="${KUBECTL:-oc}"

echo "Removing latency fault injection manifests…"
$KUBECTL delete -f "$FIXTURE_DIR/manifests.yaml" --ignore-not-found

echo "Removing any ratings VirtualService or DestinationRule created by the agent…"
$KUBECTL delete virtualservice  ratings -n "$NAMESPACE" --ignore-not-found || true
$KUBECTL delete destinationrule ratings -n "$NAMESPACE" --ignore-not-found || true

echo "Waiting ${WAIT_SECONDS}s for Istio metrics to stabilise after fault removal…"
sleep "$WAIT_SECONDS"
echo "Cleanup complete — latency fault injection and agent-created resources removed."
