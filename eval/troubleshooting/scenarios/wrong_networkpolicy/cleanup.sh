#!/usr/bin/env bash
set -euo pipefail

NS="service-mesh"
oc delete deployment frontend -n "$NS" --ignore-not-found
oc delete deployment backend -n "$NS" --ignore-not-found
oc delete svc backend-service -n "$NS" --ignore-not-found
oc delete networkpolicy backend-network-policy -n "$NS" --ignore-not-found
oc delete namespace "$NS" --ignore-not-found
