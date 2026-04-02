#!/usr/bin/env bash
set -euo pipefail

FIXTURE_DIR="$(cd "$(dirname "$0")/fixtures" && pwd)"

oc delete -f "$FIXTURE_DIR/manifests.yaml" --ignore-not-found
oc delete namespace fleet-alpha fleet-alpha1 --ignore-not-found
