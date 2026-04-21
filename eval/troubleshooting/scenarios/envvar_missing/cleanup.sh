#!/usr/bin/env bash
set -euo pipefail

oc delete -f "$(cd "$(dirname "$0")/fixtures" && pwd)/deployment.yaml" --ignore-not-found --wait=false
oc delete namespace warehouse-ops --ignore-not-found
