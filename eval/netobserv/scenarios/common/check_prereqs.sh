#!/usr/bin/env bash
# Shared prerequisites and helpers for NetObserv troubleshooting eval scenarios.
# Requires NetObserv (FlowCollector) and observability MCP tools (obs-mcp, optional kubernetes-mcp-server netobserv).

set -euo pipefail

NETOBSERV_NS="${NETOBSERV_NS:-netobserv}"
TARGET_NS="${TARGET_NS:-netobserv}"

# Optional space-separated FlowCollector eBPF features (e.g. "DNSTracking PacketDrop").
REQUIRED_NETOBSERV_FEATURES="${REQUIRED_NETOBSERV_FEATURES:-}"

check_netobserv_feature() {
  local feature="$1"
  local fc_json="$2"
  if echo "${fc_json}" | jq -e --arg f "${feature}" '
    [.items[0].spec.agent.ebpf.features[]? | if type == "string" then . else .name end] | index($f)
  ' >/dev/null; then
    return 0
  fi
  echo "WARN: FlowCollector does not list eBPF feature '${feature}' — NetObserv may not export related metrics/flows"
  return 1
}

check_netobserv_prereqs() {
  if ! command -v oc >/dev/null 2>&1; then
    echo "ERROR: oc is required for NetObserv scenarios"
    exit 1
  fi

  if ! oc whoami >/dev/null 2>&1; then
    echo "ERROR: not logged in to OpenShift (oc whoami failed)"
    exit 1
  fi

  if ! oc api-resources --api-group=flows.netobserv.io 2>/dev/null | grep -q flowcollectors; then
    echo "ERROR: FlowCollector CRD (flows.netobserv.io) not found — install NetObserv operator first"
    exit 1
  fi

  FC_JSON="$(oc get flowcollector -A -o json 2>/dev/null || true)"
  if [[ -z "${FC_JSON}" || "$(echo "${FC_JSON}" | jq '.items | length')" == "0" ]]; then
    echo "ERROR: no FlowCollector resource found in the cluster"
    exit 1
  fi

  FC_NAME="$(echo "${FC_JSON}" | jq -r '.items[0].metadata.name')"
  FC_READY="$(echo "${FC_JSON}" | jq -r '.items[0].status.conditions[]? | select(.type=="Ready") | .status' | head -n1)"

  echo "FlowCollector: ${FC_NAME} (Ready=${FC_READY:-unknown})"

  if [[ "${FC_READY}" != "True" ]]; then
    echo "WARN: FlowCollector is not Ready — NetObserv metrics/logs may be incomplete"
  fi

  if ! oc get namespace "${TARGET_NS}" >/dev/null 2>&1; then
    echo "WARN: target namespace '${TARGET_NS}' does not exist — agent may still answer cluster-wide"
  fi

  if [[ -n "${REQUIRED_NETOBSERV_FEATURES}" ]]; then
    for feat in ${REQUIRED_NETOBSERV_FEATURES}; do
      check_netobserv_feature "${feat}" "${FC_JSON}" || true
    done
  fi

  echo "NetObserv prerequisites OK (TARGET_NS=${TARGET_NS}, NETOBSERV_NS=${NETOBSERV_NS})"
  echo "OLS must have MCP access to Prometheus/Thanos (obs-mcp) and optionally NetObserv console or Loki tools."
}

# OpenShift restricted SCC requires runAsUser inside the namespace UID allocation.
openshift_namespace_uid_min() {
  local ns="$1"
  local uid_range
  uid_range="$(oc get namespace "${ns}" -o jsonpath='{.metadata.annotations.openshift\.io/sa\.scc\.uid-range}' 2>/dev/null || true)"
  if [[ -z "${uid_range}" || "${uid_range}" != */* ]]; then
    return 1
  fi
  echo "${uid_range%%/*}"
}

# Patch busybox/python/iperf containers to the namespace min UID; skip images with a fixed non-root USER.
patch_openshift_deployments() {
  local ns="$1"
  local uid_min
  if ! uid_min="$(openshift_namespace_uid_min "${ns}")"; then
    echo "No OpenShift UID range on ${ns} — using manifest runAsUser values"
    return 0
  fi

  echo "OpenShift: patching deployments in ${ns} to runAsUser=${uid_min}"
  local deploy
  for deploy in $(oc get deploy -n "${ns}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
    [[ -z "${deploy}" ]] && continue
    local json container_count i
    json="$(oc get deploy "${deploy}" -n "${ns}" -o json)"
    container_count="$(echo "${json}" | jq '.spec.template.spec.containers | length')"
    i=0
    while [[ "${i}" -lt "${container_count}" ]]; do
      local image
      image="$(echo "${json}" | jq -r ".spec.template.spec.containers[${i}].image")"
      case "${image}" in
        *nginx-unprivileged* | *curlimages/curl*)
          i=$((i + 1))
          continue
          ;;
      esac
      oc patch deploy "${deploy}" -n "${ns}" --type=json \
        -p="[{\"op\":\"add\",\"path\":\"/spec/template/spec/containers/${i}/securityContext/runAsUser\",\"value\":${uid_min}}]" \
        2>/dev/null \
        || oc patch deploy "${deploy}" -n "${ns}" --type=json \
          -p="[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/${i}/securityContext/runAsUser\",\"value\":${uid_min}}]"
      i=$((i + 1))
    done
  done
}

wait_for_namespace_gone() {
  local ns="$1"
  local max_attempts="${2:-90}"
  local attempt
  for attempt in $(seq 1 "${max_attempts}"); do
    if ! oc get namespace "${ns}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "WARN: namespace ${ns} still terminating after $((max_attempts * 2))s"
  return 1
}

deploy_netobserv_fixture() {
  local fixture_dir="$1"
  local ns="$2"
  local recreate="${NETOBSERV_EVAL_RECREATE_NS:-true}"

  if [[ "${recreate}" == "true" ]] && oc get namespace "${ns}" >/dev/null 2>&1; then
    echo "Recreating eval namespace ${ns} for a clean fixture deploy…"
    oc delete namespace "${ns}" --ignore-not-found --wait=false
    wait_for_namespace_gone "${ns}" || true
  fi

  oc apply -f "${fixture_dir}/manifest.yaml"

  local attempt
  for attempt in $(seq 1 15); do
    if openshift_namespace_uid_min "${ns}" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  patch_openshift_deployments "${ns}"

  echo "Deployed fixture in namespace ${ns}"
}

cleanup_netobserv_fixture() {
  local ns="$1"
  if oc get namespace "${ns}" >/dev/null 2>&1; then
    oc delete namespace "${ns}" --ignore-not-found --wait=false
    echo "Deleted namespace ${ns}"
  else
    echo "Namespace ${ns} not present — nothing to clean up"
  fi
}
