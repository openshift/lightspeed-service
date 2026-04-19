#!/usr/bin/env bash
set -euo pipefail

FIXTURE_DIR="$(cd "$(dirname "$0")/fixtures" && pwd)"
NS_A="fleet-alpha"
NS_B="fleet-alpha1"
EXPECTED_A=6
EXPECTED_B=9

oc apply -f "$FIXTURE_DIR/manifests.yaml"

# Wait for all pods to reach Running in both namespaces
ATTEMPT=0
until [ "$ATTEMPT" -ge 20 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  COUNT_A=$(oc get pods -n "$NS_A" --no-headers --field-selector=status.phase=Running 2>/dev/null | wc -l | tr -d ' ')
  COUNT_B=$(oc get pods -n "$NS_B" --no-headers --field-selector=status.phase=Running 2>/dev/null | wc -l | tr -d ' ')

  if [ "$COUNT_A" -eq "$EXPECTED_A" ] && [ "$COUNT_B" -eq "$EXPECTED_B" ]; then
    echo "Pod counts OK — $NS_A:$COUNT_A  $NS_B:$COUNT_B"
    exit 0
  fi
  echo "attempt $ATTEMPT/20 — $NS_A:$COUNT_A/$EXPECTED_A  $NS_B:$COUNT_B/$EXPECTED_B — waiting 3s…"
  sleep 3
done

echo "Expected pod counts not reached"
oc get pods -n "$NS_A"
oc get pods -n "$NS_B"
exit 1
