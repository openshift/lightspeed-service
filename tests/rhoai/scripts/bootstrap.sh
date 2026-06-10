#!/bin/bash

set -euo pipefail

BASE_DIR="$1"

wait_for_operator() {
  local OPERATOR_LABEL=$1
  local NAMESPACE=$2
  local OPERATOR_NAME=$3

  echo "  -> Waiting for ${OPERATOR_NAME} CSV resource to be created in namespace ${NAMESPACE}..."
  until oc get csv -n "${NAMESPACE}" -l "${OPERATOR_LABEL}" --no-headers 2>/dev/null | grep -q .; do
    echo "     ...still waiting for ${OPERATOR_NAME} CSV to show up"
    sleep 5
  done

  echo "  -> Waiting for ${OPERATOR_NAME} CSV to reach Succeeded..."
  oc wait --for=jsonpath='{.status.phase}'=Succeeded csv -n "${NAMESPACE}" -l "${OPERATOR_LABEL}" --timeout=600s
}

# APPLY OPERATOR SUBSCRIPTIONS
echo "--> Applying OperatorGroups from operatorgroup.yaml..."
oc apply -f "$BASE_DIR/manifests/operators/operatorgroup.yaml"

sleep 10

echo "--> Applying Operator Subscriptions from operators.yaml..."
oc apply -f "$BASE_DIR/manifests/operators/operators.yaml"

sleep 10

# WAIT FOR GPU OPERATOR NAMESPACE AND OPERATORGROUP
echo "--> Ensuring GPU Operator namespace and OperatorGroup are ready..."
oc wait --for=jsonpath='{.status.phase}'=Active namespace/nvidia-gpu-operator --timeout=60s
echo "  -> Waiting for GPU OperatorGroup to be created..."
until oc get operatorgroup nvidia-gpu-operator-group -n nvidia-gpu-operator &>/dev/null; do
  echo "     ...still waiting for OperatorGroup"
  sleep 2
done
echo "  -> GPU OperatorGroup ready"

# Give OLM a moment to process the OperatorGroup before checking subscriptions
sleep 5

# WAIT FOR OPERATORS TO BECOME READY
echo "--> Waiting for Operators to be installed. This can take several minutes..."

# Ensure the ClusterServiceVersion CRD exists before checking for CSVs
oc wait --for=condition=established --timeout=300s crd/clusterserviceversions.operators.coreos.com

wait_for_operator "operators.coreos.com/rhods-operator.openshift-operators" "openshift-operators" "RHODS Operator"

# Verify GPU operator InstallPlan was created before waiting for CSV
echo "  -> Verifying GPU Operator InstallPlan was created..."
timeout=120
elapsed=0
until oc get installplan -n nvidia-gpu-operator --no-headers 2>/dev/null | grep -q .; do
  if [ $elapsed -ge $timeout ]; then
    echo "     ❌ No InstallPlan created for GPU Operator - this is an OLM issue"
    echo "     Attempting to fix by recreating subscription..."
    oc delete subscription gpu-operator-certified -n nvidia-gpu-operator
    sleep 5
    oc apply -f "$BASE_DIR/manifests/operators/operators.yaml"
    sleep 10
    # Try one more time
    if ! oc get installplan -n nvidia-gpu-operator --no-headers 2>/dev/null | grep -q .; then
      echo "     ❌ Still no InstallPlan - manual intervention required"
      exit 1
    fi
    break
  fi
  echo "     ...waiting for InstallPlan ($elapsed/$timeout seconds)"
  sleep 5
  elapsed=$((elapsed + 5))
done
echo "  -> InstallPlan created successfully"

wait_for_operator "operators.coreos.com/gpu-operator-certified.nvidia-gpu-operator" "nvidia-gpu-operator" "GPU Operator"
wait_for_operator "operators.coreos.com/nfd.openshift-nfd" "openshift-nfd" "NFD Operator"

echo "  -> Waiting for NFD CRD to be established..."
oc wait --for=condition=established --timeout=300s crd/nodefeaturediscoveries.nfd.openshift.io

echo "--> All operators are ready."

oc get csv -n openshift-operators
oc get csv -n nvidia-gpu-operator
oc get csv -n openshift-nfd

echo "--> Applying DataScienceCluster from ds-cluster.yaml..."
oc apply -f "$BASE_DIR/manifests/operators/ds-cluster.yaml"
sleep 5
sleep 10

echo "--> Checking DSCInitialization and DSC status..."
oc get dsci -A -o jsonpath='{range .items[*]}DSCI: {.metadata.name} applicationsNS: {.spec.applicationsNamespace}{"\n"}{end}' 2>/dev/null || echo "No DSCInitialization found"
oc get dsc -A -o jsonpath='{range .items[*]}DSC: {.metadata.name} phase: {.status.phase}{"\n"}{end}' 2>/dev/null || echo "No DSC status yet"

echo "All files applied successfully. The DataScienceCluster is now provisioning."
