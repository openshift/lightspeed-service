#!/bin/bash

set -euo pipefail

BASE_DIR="$1"

echo "Setting up GPU support..."

# Debug: Show all nodes and their instance types
echo ""
echo "--> DEBUG: Cluster nodes before GPU setup..."
oc get nodes -o custom-columns=NAME:.metadata.name,INSTANCE:.metadata.labels.node\\.kubernetes\\.io/instance-type,STATUS:.status.conditions[-1].type

# Debug: Check for GPU instance types and taints
echo ""
echo "--> DEBUG: Checking for GPU nodes and taints..."
gpu_nodes=$(oc get nodes -o jsonpath='{range .items[*]}{.metadata.name}{","}{.metadata.labels.node\.kubernetes\.io/instance-type}{"\n"}{end}' | grep -E "g4dn|p3|p4|g5" | cut -d',' -f1 || echo "")

if [ -n "$gpu_nodes" ]; then
  echo "    Found GPU instance types:"
  for node in $gpu_nodes; do
    echo "    Node: $node"
    echo "    Instance Type: $(oc get node $node -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}')"
    echo "    Taints:"
    oc get node $node -o jsonpath='{.spec.taints}' || echo "      No taints"
    echo ""
  done
else
  echo "    No GPU instance types found (g4dn, p3, p4, g5)"
  echo "    All node instance types:"
  oc get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.labels.node\.kubernetes\.io/instance-type}{"\n"}{end}'
fi

# Apply NFD instance
echo ""
echo "--> Applying NFD instance..."
oc apply -f "$BASE_DIR/manifests/gpu/create-nfd.yaml"

# Wait for NFD worker daemonset to be created
echo "--> Waiting for NFD worker daemonset to be created..."
timeout=60
elapsed=0
until oc get daemonset nfd-worker -n openshift-nfd &>/dev/null; do
  if [ $elapsed -ge $timeout ]; then
    echo "❌ Timeout waiting for NFD worker daemonset"
    exit 1
  fi
  echo "   Waiting for nfd-worker daemonset... ($elapsed/$timeout seconds)"
  sleep 5
  elapsed=$((elapsed + 5))
done
echo "✅ NFD worker daemonset created"

# Patch NFD worker daemonset to add GPU node tolerations
# This is needed in the prow env to be assigned a GPU
echo "--> Patching NFD worker daemonset with GPU tolerations..."
oc patch daemonset nfd-worker -n openshift-nfd --type=json -p='[
  {
    "op": "add",
    "path": "/spec/template/spec/tolerations",
    "value": [
      {
        "key": "nvidia.com/gpu",
        "operator": "Exists",
        "effect": "NoSchedule"
      },
      {
        "key": "gpu",
        "operator": "Exists",
        "effect": "NoSchedule"
      }
    ]
  }
]'
echo "✅ NFD worker tolerations added"

# Apply ClusterPolicy
echo ""
echo "--> Applying ClusterPolicy..."
oc apply -f "$BASE_DIR/manifests/gpu/cluster-policy.yaml"

# Wait for GPU operator pods to be created and healthy
echo ""
echo "--> Waiting for GPU operator pods to be healthy..."
echo "    This may take up to 10 minutes while images are pulled and pods start..."
timeout=1200
elapsed=0
until oc get pods -n nvidia-gpu-operator --no-headers 2>/dev/null | awk '{if ($3 != "Running" && $3 != "Completed") exit 1}' && [ $(oc get pods -n nvidia-gpu-operator --no-headers 2>/dev/null | wc -l) -gt 5 ]; do
  if [ $elapsed -ge $timeout ]; then
    echo "❌ Timeout waiting for GPU operator pods to be healthy"
    echo "Current pod status:"
    oc get pods -n nvidia-gpu-operator
    echo ""
    echo "DEBUG: Checking for scheduling issues..."
    oc get pods -n nvidia-gpu-operator -o wide
    echo ""
    echo "DEBUG: Checking pod events for failures..."
    oc get events -n nvidia-gpu-operator --sort-by='.lastTimestamp' | tail -20
    exit 1
  fi
  pod_count=$(oc get pods -n nvidia-gpu-operator --no-headers 2>/dev/null | wc -l || echo 0)
  failed_pods=$(oc get pods -n nvidia-gpu-operator --no-headers 2>/dev/null | awk '{if ($3 != "Running" && $3 != "Completed") print $1}' | wc -l || echo 0)
  echo "   Pods: $pod_count total, $failed_pods not ready. Waiting... ($elapsed/$timeout seconds)"

  # Show additional debug info every 60 seconds
  if [ $((elapsed % 60)) -eq 0 ] && [ $elapsed -gt 0 ]; then
    echo "   DEBUG: Current pod statuses:"
    oc get pods -n nvidia-gpu-operator -o wide
  fi

  sleep 15
  elapsed=$((elapsed + 15))
done
echo "✅ All GPU operator pods are healthy"

# Debug: Show what pods are running
echo ""
echo "--> DEBUG: GPU operator pods deployed:"
oc get pods -n nvidia-gpu-operator -o wide

# Wait for GPU nodes to be labeled by NFD
echo ""
echo "--> Waiting for GPU nodes to be labeled by NFD..."
timeout=120
elapsed=0
until oc get nodes -l nvidia.com/gpu.present=true --no-headers 2>/dev/null | grep -q .; do
  if [ $elapsed -ge $timeout ]; then
    echo "❌ Timeout waiting for GPU nodes to be labeled"
    echo ""
    echo "DEBUG: Checking why nodes aren't labeled..."
    echo "All node labels related to features:"
    oc get nodes --show-labels | grep -E "feature|gpu|nvidia" || echo "No GPU/feature labels found on any nodes"
    echo ""
    echo "DEBUG: NFD worker pods status:"
    oc get pods -n openshift-nfd -o wide
    echo ""
    echo "DEBUG: Recent NFD events:"
    oc get events -n openshift-nfd --sort-by='.lastTimestamp' | tail -10
    exit 1
  fi
  echo "   No GPU nodes found yet. Waiting... ($elapsed/$timeout seconds)"

  # Show debug info every 30 seconds
  if [ $((elapsed % 30)) -eq 0 ] && [ $elapsed -gt 0 ]; then
    echo "   DEBUG: Checking NFD worker pods..."
    oc get pods -n openshift-nfd --no-headers
  fi

  sleep 10
  elapsed=$((elapsed + 10))
done
echo "✅ GPU nodes detected"

# Debug: Show labeled nodes
echo ""
echo "--> DEBUG: GPU nodes labeled:"
oc get nodes -l nvidia.com/gpu.present=true -o custom-columns=NAME:.metadata.name,INSTANCE:.metadata.labels.node\\.kubernetes\\.io/instance-type,TAINTS:.spec.taints

# Wait for GPU capacity AND allocatable to become available
echo "--> Waiting for GPU capacity and allocatable to be available on nodes..."
timeout=120
elapsed=0
until [ "$(oc get nodes -l nvidia.com/gpu.present=true -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}' 2>/dev/null)" != "" ] && \
      [ "$(oc get nodes -l nvidia.com/gpu.present=true -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}' 2>/dev/null)" != "0" ]; do
  if [ $elapsed -ge $timeout ]; then
    echo "❌ Timeout waiting for GPU capacity/allocatable"
    echo ""
    echo "DEBUG: Investigating why GPU capacity is not appearing..."
    echo "Device plugin pods:"
    oc get pods -n nvidia-gpu-operator -l app=nvidia-device-plugin-daemonset -o wide
    echo ""
    echo "Device plugin pod logs (last 20 lines):"
    device_plugin_pod=$(oc get pods -n nvidia-gpu-operator -l app=nvidia-device-plugin-daemonset -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    if [ -n "$device_plugin_pod" ]; then
      oc logs -n nvidia-gpu-operator "$device_plugin_pod" --tail=20 || echo "Could not fetch logs"
    else
      echo "No device plugin pod found - checking for scheduling issues..."
      oc get events -n nvidia-gpu-operator --field-selector involvedObject.name=nvidia-device-plugin-daemonset --sort-by='.lastTimestamp' | tail -10
    fi
    echo ""
    echo "Node GPU capacity details:"
    oc get nodes -l nvidia.com/gpu.present=true -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{"capacity: "}{.status.capacity.nvidia\.com/gpu}{"\t"}{"allocatable: "}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
    exit 1
  fi
  capacity=$(oc get nodes -l nvidia.com/gpu.present=true -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}' 2>/dev/null || echo "0")
  allocatable=$(oc get nodes -l nvidia.com/gpu.present=true -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "0")
  echo "   GPU capacity: $capacity, allocatable: $allocatable. Waiting for both > 0... ($elapsed/$timeout seconds)"

  # Show debug info every 30 seconds
  if [ $((elapsed % 30)) -eq 0 ] && [ $elapsed -gt 0 ]; then
    echo "   DEBUG: Checking device plugin daemonset pods..."
    oc get pods -n nvidia-gpu-operator -l app=nvidia-device-plugin-daemonset --no-headers
  fi

  sleep 15
  elapsed=$((elapsed + 15))
done

echo ""
echo "✅ GPU setup complete!"
echo ""
echo "GPU Node Status:"
oc get nodes -l nvidia.com/gpu.present=true -o custom-columns=NAME:.metadata.name,GPU:.status.capacity.nvidia\\.com/gpu,ALLOCATABLE:.status.allocatable.nvidia\\.com/gpu,INSTANCE:.metadata.labels.node\\.kubernetes\\.io/instance-type

echo ""
echo "ClusterPolicy Status:"
oc get clusterpolicy gpu-cluster-policy -o jsonpath='{.status.state}'
echo ""
