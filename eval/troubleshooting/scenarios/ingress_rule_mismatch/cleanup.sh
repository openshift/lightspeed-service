#!/usr/bin/env bash
set -euo pipefail

NS="platform-core"
oc delete deployment web-portal -n "$NS" --ignore-not-found
oc delete deployment api-gateway -n "$NS" --ignore-not-found
oc delete svc api-gateway-svc -n "$NS" --ignore-not-found
oc delete networkpolicy restrict-api-gateway-ingress -n "$NS" --ignore-not-found
oc delete namespace "$NS" --ignore-not-found
