#!/usr/bin/env bash
set -euo pipefail

oc delete namespace catalog-mgmt --ignore-not-found
