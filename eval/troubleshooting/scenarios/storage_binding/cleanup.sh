#!/usr/bin/env bash
set -euo pipefail

NS="cache-tier"
oc delete pvc memcached-data-pvc -n "$NS" --ignore-not-found
oc delete deployment memcached -n "$NS" --ignore-not-found
oc delete svc memcached -n "$NS" --ignore-not-found
oc delete namespace "$NS" --ignore-not-found
