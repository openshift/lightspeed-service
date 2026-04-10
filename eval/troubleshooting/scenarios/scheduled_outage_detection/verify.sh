#!/usr/bin/env bash
set -euo pipefail

NS="analytics-platform"
APP="report-generator"

LOGS=$(oc logs -l "app=$APP" -n "$NS" --tail=10000 2>/dev/null)
[ -n "$LOGS" ] || { echo "FAIL: no logs from $APP"; exit 1; }

for sentinel in \
  "Detected repeated failures during 03:00-03:05 window" \
  "System health check passed" \
  "Job executed successfully in 167ms."; do
  echo "$LOGS" | grep "$sentinel" >/dev/null \
    || { echo "FAIL: missing sentinel — $sentinel"; exit 1; }
done

echo "PASS: $APP has all expected log sentinels"
