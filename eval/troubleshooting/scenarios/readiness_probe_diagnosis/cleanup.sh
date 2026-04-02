#!/usr/bin/env bash
set -euo pipefail

oc delete namespace discovery-hub --ignore-not-found --wait=false
