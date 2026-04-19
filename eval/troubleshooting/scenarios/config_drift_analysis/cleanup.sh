#!/usr/bin/env bash
set -euo pipefail

NS="ingress-layer"

oc delete deployment gateway-proxy -n "$NS" --ignore-not-found --wait=false
oc delete secret gateway-proxy-log-script -n "$NS" --ignore-not-found
oc delete namespace "$NS" --ignore-not-found
