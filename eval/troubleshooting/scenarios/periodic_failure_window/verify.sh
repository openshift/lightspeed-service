#!/usr/bin/env bash
set -euo pipefail

NS="data-pipeline"
APP="batch-processor"

LOGS=$(oc logs -l "app=$APP" -n "$NS" --tail=10000 2>/dev/null)
[ -n "$LOGS" ] || { echo "FAIL: no logs from $APP"; exit 1; }

MISSING=()
for sentinel in \
  "Detected repeated failures during 03:00-03:05 window" \
  "System health check passed" \
  "Job executed successfully in 167ms."; do
  echo "$LOGS" | grep -F "$sentinel" >/dev/null || MISSING+=("$sentinel")
done

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "FAIL: missing ${#MISSING[@]} sentinel(s):"
  printf '  - %s\n' "${MISSING[@]}"
  exit 1
fi

echo "PASS: $APP has all expected log sentinels"
