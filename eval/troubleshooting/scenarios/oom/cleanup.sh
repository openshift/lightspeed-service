#!/usr/bin/env bash
set -euo pipefail

NS="oom-scenario"
oc delete deployment awesome-application -n "$NS" --ignore-not-found
oc delete namespace "$NS" --ignore-not-found
