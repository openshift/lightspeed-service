#!/usr/bin/env bash
set -euo pipefail

NS="data-pipeline"

oc delete deployment batch-processor -n "$NS" --ignore-not-found --grace-period=0
oc delete secret batch-processor-logs-script -n "$NS" --ignore-not-found
oc delete namespace "$NS" --ignore-not-found
