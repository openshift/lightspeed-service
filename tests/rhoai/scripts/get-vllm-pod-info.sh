#!/bin/bash

NAMESPACE="e2e-rhoai-dsc"
ISVC_NAME="${1:-vllm-model}"
ENV_FILE="${ENV_FILE:-pod.env}"

KSVC_NAME="${ISVC_NAME}-predictor"

echo "--> Finding the pod for InferenceService '$ISVC_NAME'..."

# Find the running pod for the InferenceService
POD_NAME=""
CURRENT_POD=""
CURRENT_STATUS=""
TIMEOUT=580
INTERVAL=20
ELAPSED=0

until [ -n "$POD_NAME" ] || [ $ELAPSED -ge $TIMEOUT ]; do
  # Get the pod name regardless of status for visibility
  CURRENT_POD=$(oc get pods -n "$NAMESPACE" \
    -l "serving.kserve.io/inferenceservice=$ISVC_NAME" \
    -o jsonpath="{.items[0].metadata.name}" 2>/dev/null)

  # Get the pod status
  CURRENT_STATUS=$(oc get pods -n "$NAMESPACE" \
    -l "serving.kserve.io/inferenceservice=$ISVC_NAME" \
    -o jsonpath="{.items[0].status.phase}" 2>/dev/null)

  # Check if a running pod exists
  POD_NAME=$(oc get pods -n "$NAMESPACE" \
    -l "serving.kserve.io/inferenceservice=$ISVC_NAME" \
    -o jsonpath="{.items[?(@.status.phase=='Running')].metadata.name}" 2>/dev/null)

  if [ -n "$CURRENT_POD" ]; then
    echo "Waiting for pod $CURRENT_POD in namespace $NAMESPACE (current status: ${CURRENT_STATUS:-Unknown})"
    # Show more debug info if pod exists but isn't Running
    if [ -z "$POD_NAME" ] && [ $((ELAPSED % 60)) -eq 0 ]; then
      echo "  DEBUG: Pod details:"
      oc get pod "$CURRENT_POD" -n "$NAMESPACE" -o wide || true
      echo "  DEBUG: Pod events:"
      oc get events -n "$NAMESPACE" --field-selector involvedObject.name="$CURRENT_POD" --sort-by='.lastTimestamp' | tail -5 || true
    fi
  else
    echo "Waiting for pod with label serving.kserve.io/inferenceservice=$ISVC_NAME in namespace $NAMESPACE (no pod found yet)"
    # Show InferenceService status if no pod found
    if [ $((ELAPSED % 60)) -eq 0 ]; then
      echo "  DEBUG: InferenceService status:"
      oc get inferenceservice "$ISVC_NAME" -n "$NAMESPACE" -o jsonpath='{.status.conditions}' || true
      echo ""
      echo "  DEBUG: All pods in namespace:"
      oc get pods -n "$NAMESPACE" || true
    fi
  fi

  if [ -z "$POD_NAME" ]; then
    echo "  -> Pod not running yet, waiting $INTERVAL seconds... ($ELAPSED/$TIMEOUT)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
  fi
done

# Exit immediately if no running pod was found
if [ -z "$POD_NAME" ]; then
  echo ""
  echo "❌ Timeout reached after $TIMEOUT seconds. Pod is not running."
  echo ""
  echo "DEBUG: InferenceService status:"
  oc describe inferenceservice "$ISVC_NAME" -n "$NAMESPACE" || true
  echo ""
  echo "DEBUG: All pods in namespace:"
  oc get pods -n "$NAMESPACE" -o wide || true
  echo ""
  echo "DEBUG: Recent events:"
  oc get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -20 || true
  exit 1
fi

echo "  -> Pod is running: $POD_NAME"

# Show pod details
oc describe pod "$POD_NAME" -n "$NAMESPACE" || true
oc logs "$POD_NAME" -n "$NAMESPACE" --tail=50 || true

# Get the 'app' label for Service selector
APP_LABEL=$(oc get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.app}')
if [ -z "$APP_LABEL" ]; then
  echo "Error: Could not find 'app' label on pod $POD_NAME"
  exit 1
fi
echo "  -> Found 'app' label: $APP_LABEL"

# Check if this is RawDeployment mode (standard K8s Service) or Serverless (Knative Service)
if oc get ksvc "$KSVC_NAME" -n "$NAMESPACE" &>/dev/null; then
  # Serverless mode - get Knative Service URL
  KSVC_URL=$(oc get ksvc "$KSVC_NAME" -n "$NAMESPACE" -o jsonpath='{.status.url}')
  echo "  -> Found Knative Service URL: $KSVC_URL"
else
  # RawDeployment mode - construct URL from standard K8s Service
  echo "  -> RawDeployment mode detected, looking for standard Kubernetes Service..."
  SERVICE_NAME="${ISVC_NAME}-predictor"

  # Check if the service exists
  if ! oc get service "$SERVICE_NAME" -n "$NAMESPACE" &>/dev/null; then
    echo "Error: Could not find Service $SERVICE_NAME"
    exit 1
  fi

  # Get the cluster IP and targetPort (the actual container port)
  CLUSTER_IP=$(oc get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
  SERVICE_PORT=$(oc get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].port}')
  TARGET_PORT=$(oc get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].targetPort}')

  # Use targetPort (container port) instead of service port for RawDeployment
  PORT=${TARGET_PORT:-$SERVICE_PORT}

  # Construct internal cluster URL
  KSVC_URL="http://${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local:${PORT}"
  echo "  -> Found Service URL: $KSVC_URL (Cluster IP: $CLUSTER_IP, Service Port: $SERVICE_PORT, Target Port: $TARGET_PORT)"
fi

# Save all info to pod.env
cat <<EOF > "$ENV_FILE"
# Environment variables for the vLLM service
POD_NAME=$POD_NAME
APP_LABEL=$APP_LABEL
NAMESPACE=$NAMESPACE
ISVC_NAME=$ISVC_NAME
KSVC_NAME=$KSVC_NAME
KSVC_URL=$KSVC_URL
EOF

echo "✅ Success! Details saved in $ENV_FILE."
