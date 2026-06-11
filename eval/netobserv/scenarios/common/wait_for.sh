#!/usr/bin/env bash
# Wait helpers for NetObserv eval scenarios.

wait_for_rollout() {
  local ns="$1"
  local deployment="$2"
  local timeout="${3:-120s}"
  oc wait --for=condition=available "deployment/${deployment}" -n "$ns" --timeout="$timeout"
}

wait_for_log_pattern() {
  local ns="$1"
  local selector="$2"
  local pattern="$3"
  local max_attempts="${4:-40}"
  local sleep_secs="${5:-3}"

  local attempt
  for attempt in $(seq 1 "$max_attempts"); do
    if oc logs -n "$ns" -l "$selector" --tail=30 2>/dev/null | grep -qE "$pattern"; then
      echo "Log pattern matched (attempt ${attempt}/${max_attempts})"
      return 0
    fi
    echo "  attempt ${attempt}/${max_attempts}: waiting for log /${pattern}/ …"
    sleep "$sleep_secs"
  done
  echo "ERROR: log pattern not found within timeout"
  oc get pods -n "$ns" -l "$selector" 2>/dev/null || true
  oc logs -n "$ns" -l "$selector" --tail=20 2>/dev/null || true
  return 1
}
