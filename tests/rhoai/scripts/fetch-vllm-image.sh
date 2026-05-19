#!/bin/bash

# Get vLLM CUDA image from RHOAI ServingRuntime template
echo "Fetching vLLM CUDA image from RHOAI..."
# Wait for RHOAI templates to be available (up to 20 minutes for first check)
timeout=1200
elapsed=0
until oc get template vllm-cuda-runtime-template -n redhat-ods-applications &>/dev/null; do
  if [ $elapsed -ge $timeout ]; then
    echo "❌ Timeout waiting for RHOAI templates (waited $timeout seconds)"
    exit 1
  fi
  echo "  -> Waiting for RHOAI templates... ($elapsed/$timeout seconds)"
  sleep 10
  elapsed=$((elapsed + 10))
done

# Extract vLLM image from the template
VLLM_IMAGE=$(oc get template vllm-cuda-runtime-template -n redhat-ods-applications -o jsonpath='{.objects[0].spec.containers[0].image}' 2>/dev/null || echo "")

# Fallback: check existing ServingRuntimes for vLLM image
if [ -z "$VLLM_IMAGE" ]; then
  echo "  -> Template not found, checking existing ServingRuntimes..."
  # Get all serving runtimes and filter for vLLM ones
  VLLM_IMAGE=$(oc get servingruntime -A -o jsonpath='{range .items[*]}{.metadata.name}{","}{.spec.containers[0].image}{"\n"}{end}' 2>/dev/null | grep -i vllm | cut -d',' -f2 | grep 'odh-vllm-cuda-rhel9' | head -1 || echo "")
fi

# Fallback: use default if still not found
if [ -z "$VLLM_IMAGE" ]; then
  echo "  -> Could not find vLLM image dynamically, using fallback..."
  VLLM_IMAGE="registry.redhat.io/rhoai/odh-vllm-cuda-rhel9@sha256:5b86924790aeb996a7e3b7f9f4c8a3a676a83cd1d7484ae584101722d362c69b"
fi
echo "  -> Found vLLM image: $VLLM_IMAGE"

# Export images as environment variables for manifest substitution
export VLLM_IMAGE
