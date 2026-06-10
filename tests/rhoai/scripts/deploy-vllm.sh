#!/bin/bash

BASE_DIR="$1"

# Wait until the CRDs exist
for crd in servingruntimes.serving.kserve.io inferenceservices.serving.kserve.io; do
  echo "Waiting for CRD $crd to exist..."
  until oc get crd $crd &>/dev/null; do
    sleep 5
  done
  echo "CRD $crd exists. Waiting to be established..."
  oc wait --for=condition=established crd/$crd --timeout=120s
done

# Wait for KServe controller deployment to appear
echo "Waiting for kserve-controller-manager deployment to be created..."
until oc get deployment kserve-controller-manager -n redhat-ods-applications &>/dev/null; do
  sleep 10
done

# Wait for rollout to complete
echo "Waiting for kserve-controller-manager rollout..."
oc rollout status deployment/kserve-controller-manager -n redhat-ods-applications --timeout=300s

# Wait for the webhook service endpoints to become ready
echo "Waiting for KServe webhook service endpoints..."
until oc get endpoints kserve-webhook-server-service -n redhat-ods-applications -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null | grep -qE '.'; do
  sleep 5
done
echo "✅ KServe webhook service is ready."

# Wait for GPU nodes to be labeled by NFD
echo "Waiting for GPU nodes to be labeled by NFD..."
timeout=600  # 10 minutes
elapsed=0
until oc get nodes -l nvidia.com/gpu.present=true --no-headers 2>/dev/null | grep -q .; do
  if [ $elapsed -ge $timeout ]; then
    echo "❌ Timeout waiting for GPU nodes to be labeled"
    exit 1
  fi
  echo "No GPU nodes found yet. Waiting... ($elapsed/$timeout seconds)"
  sleep 10
  elapsed=$((elapsed + 10))
done
echo "✅ GPU nodes detected."

# Wait for GPU capacity to be available
echo "Waiting for GPU capacity to be available on nodes..."
timeout=600  # 10 minutes
elapsed=0
until [ "$(oc get nodes -l nvidia.com/gpu.present=true -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}' 2>/dev/null)" != "" ] && \
      [ "$(oc get nodes -l nvidia.com/gpu.present=true -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}' 2>/dev/null)" != "0" ]; do
  if [ $elapsed -ge $timeout ]; then
    echo "❌ Timeout waiting for GPU capacity"
    echo "DEBUG: Checking GPU status..."
    oc get nodes -l nvidia.com/gpu.present=true -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{"capacity: "}{.status.capacity.nvidia\.com/gpu}{"\t"}{"allocatable: "}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
    exit 1
  fi
  capacity=$(oc get nodes -l nvidia.com/gpu.present=true -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}' 2>/dev/null || echo "0")
  echo "GPU capacity: $capacity. Waiting... ($elapsed/$timeout seconds)"
  sleep 10
  elapsed=$((elapsed + 10))
done
echo "✅ GPU capacity available."

# Display GPU node info
echo "GPU nodes ready:"
oc get nodes -l nvidia.com/gpu.present=true -o custom-columns=NAME:.metadata.name,GPU:.status.capacity.nvidia\\.com/gpu,INSTANCE:.metadata.labels.node\\.kubernetes\\.io/instance-type

echo "Applying vLLM manifests..."

envsubst < "$BASE_DIR/manifests/vllm/vllm-runtime-gpu.yaml" | oc apply -f -

# Wait a moment for the ServingRuntime to be fully persisted before creating the InferenceService
echo "Waiting for ServingRuntime to be ready..."
sleep 5

oc apply -f "$BASE_DIR/manifests/vllm/vllm-inference-service-gpu.yaml"