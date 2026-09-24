#!/bin/bash
# Daily short-dataset LSEval: same provider/model matrix as presubmit, with
# separate suite IDs, historical score CSV, and plots in ARTIFACT_DIR.
# Requires the same credentials and OLS_IMAGE as test-lseval-presubmit.sh.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LSEVAL_RUN_KIND=daily
exec "$DIR/test-lseval-presubmit.sh"
