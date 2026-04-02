#!/usr/bin/env bash
set -euo pipefail

NS="analytics-platform"

oc delete statefulset report-generator -n "$NS" --ignore-not-found --wait=false
oc delete secret report-generator-logs-script -n "$NS" --ignore-not-found
oc delete namespace "$NS" --ignore-not-found
