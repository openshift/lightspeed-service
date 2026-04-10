#!/usr/bin/env bash
set -euo pipefail

NS_A="fleet-alpha"
NS_B="fleet-alpha1"

# Use JSON output + python to get an exact running-pod count
count_running() {
  oc get pods -n "$1" -o json 2>/dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(sum(1 for p in d['items'] if p['status'].get('phase')=='Running'))"
}

A=$(count_running "$NS_A")
B=$(count_running "$NS_B")

[ "$A" -eq 6 ] || { echo "FAIL: expected 6 running pods in $NS_A, got $A"; exit 1; }
[ "$B" -eq 9 ] || { echo "FAIL: expected 9 running pods in $NS_B, got $B"; exit 1; }

echo "PASS: $NS_A=$A pods, $NS_B=$B pods"
