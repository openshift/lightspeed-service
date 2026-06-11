#!/usr/bin/env bash
# Wait helpers for NetObserv eval scenarios.

wait_for_rollout() {
  local ns="$1"
  local deployment="$2"
  local timeout="${3:-120s}"
  if oc wait --for=condition=available "deployment/${deployment}" -n "$ns" --timeout="$timeout"; then
    return 0
  fi
  echo "ERROR: deployment/${deployment} not available in ${ns} within ${timeout}"
  oc get pods -n "$ns" -l "app=${deployment}" -o wide 2>/dev/null \
    || oc get pods -n "$ns" -o wide 2>/dev/null || true
  local pod
  for pod in $(oc get pods -n "$ns" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
    echo "--- oc describe pod/${pod} -n ${ns} (last events) ---"
    oc describe pod "${pod}" -n "$ns" 2>/dev/null | tail -25 || true
  done
  return 1
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

# Require at least min_count log lines matching pattern (sustained traffic before NetObserv export).
wait_for_min_log_matches() {
  local ns="$1"
  local selector="$2"
  local pattern="$3"
  local min_count="${4:-3}"
  local max_attempts="${5:-50}"
  local sleep_secs="${6:-3}"

  local attempt count
  for attempt in $(seq 1 "$max_attempts"); do
    count="$(oc logs -n "$ns" -l "$selector" --tail=80 2>/dev/null | grep -cE "$pattern" || true)"
    if [[ "${count}" -ge "${min_count}" ]]; then
      echo "Log pattern matched ${count} times (need >= ${min_count}, attempt ${attempt}/${max_attempts})"
      return 0
    fi
    echo "  attempt ${attempt}/${max_attempts}: ${count}/${min_count} log lines matching /${pattern}/ …"
    sleep "$sleep_secs"
  done
  echo "ERROR: fewer than ${min_count} log lines matched /${pattern}/ within timeout"
  oc logs -n "$ns" -l "$selector" --tail=30 2>/dev/null || true
  return 1
}

# NetObserv eBPF → flowlogs-pipeline → Prometheus/Loki needs time after traffic starts.
# Override with NETOBSERV_WARMUP_SECS (default 120).
wait_for_netobserv_warmup() {
  local secs="${NETOBSERV_WARMUP_SECS:-120}"
  echo "Waiting ${secs}s for NetObserv to export flows/metrics (NETOBSERV_WARMUP_SECS)…"
  sleep "$secs"
  echo "NetObserv warmup complete"
}
