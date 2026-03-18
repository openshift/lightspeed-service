#!/bin/bash

set -e

DEFAULT_BASE_IMAGE="registry.redhat.io/rhai/base-image-cpu-rhel9:3.3"
BUILD_ARGS_FILE="build.args"
INPUT_FILE="rpms.in.yaml"
OUTPUT_FILE="rpms.lock.yaml"
CONTAINER_IMAGE="registry.access.redhat.com/ubi9/ubi"
REDHAT_REPO_FILE="redhat.repo"

if command -v podman &>/dev/null; then
    CONTAINER_RUNTIME="podman"
elif command -v docker &>/dev/null; then
    CONTAINER_RUNTIME="docker"
else
    echo "Error: Neither podman nor docker found. Please install one of them."
    exit 1
fi

if [[ -f "$BUILD_ARGS_FILE" ]]; then
    EXTRACTED_BASE_IMAGE=$(grep "^BUILDER_BASE_IMAGE=" "$BUILD_ARGS_FILE" | cut -d'=' -f2)
    if [[ -n "$EXTRACTED_BASE_IMAGE" ]]; then
        BASE_IMAGE="$EXTRACTED_BASE_IMAGE"
        echo "Using base image from $BUILD_ARGS_FILE: $BASE_IMAGE"
    else
        BASE_IMAGE="$DEFAULT_BASE_IMAGE"
        echo "BUILDER_BASE_IMAGE not found in $BUILD_ARGS_FILE, using default: $BASE_IMAGE"
    fi
else
    BASE_IMAGE="$DEFAULT_BASE_IMAGE"
    echo "$BUILD_ARGS_FILE not found, using default base image: $BASE_IMAGE"
fi

usage() {
    echo "Usage: $0 -a ACTIVATION_KEY -g ORG_ID [-i BASE_IMAGE] [-f INPUT_FILE] [-O OUTPUT_FILE]"
    echo ""
    echo "Required:"
    echo "  -a ACTIVATION_KEY  Red Hat activation key for subscription-manager"
    echo "  -g ORG_ID          Red Hat organization ID for subscription-manager"
    echo ""
    echo "Options:"
    echo "  -i BASE_IMAGE      Base container image for rpm-lockfile-prototype (default: $BASE_IMAGE)"
    echo "  -f INPUT_FILE      Input RPM specification file (default: $INPUT_FILE)"
    echo "  -O OUTPUT_FILE     Output lock file (default: $OUTPUT_FILE)"
    echo "  -h                 Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  REGISTRY_USERNAME  Username for registry authentication (skopeo login)"
    echo "  REGISTRY_PASSWORD  Password for registry authentication (skopeo login)"
    exit 1
}

ACTIVATION_KEY=""
ORG_ID=""

while getopts "a:g:i:f:O:h" opt; do
    case $opt in
        a) ACTIVATION_KEY="$OPTARG" ;;
        g) ORG_ID="$OPTARG" ;;
        i) BASE_IMAGE="$OPTARG" ;;
        f) INPUT_FILE="$OPTARG" ;;
        O) OUTPUT_FILE="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ -z "$ACTIVATION_KEY" || -z "$ORG_ID" ]]; then
    echo "Error: Both activation key (-a) and organization ID (-g) are required."
    usage
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file '$INPUT_FILE' not found."
    exit 1
fi

if [[ ! -f "$REDHAT_REPO_FILE" ]]; then
    echo "Error: Red Hat repo file '$REDHAT_REPO_FILE' not found."
    exit 1
fi

echo "Using BASE_IMAGE: $BASE_IMAGE"
echo "Using INPUT_FILE: $INPUT_FILE"
echo "Using OUTPUT_FILE: $OUTPUT_FILE"
echo "Using CONTAINER_IMAGE: $CONTAINER_IMAGE"
echo "Using CONTAINER_RUNTIME: $CONTAINER_RUNTIME"

CONTAINER_NAME="rpm-lockfile-generator-$$"
WORKDIR="/workdir"

cleanup() {
    echo "Cleaning up container..."
    $CONTAINER_RUNTIME rm -f "$CONTAINER_NAME" 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting container..."
$CONTAINER_RUNTIME run -d --name "$CONTAINER_NAME" "$CONTAINER_IMAGE" sleep infinity

echo "Registering system with subscription-manager..."
$CONTAINER_RUNTIME exec "$CONTAINER_NAME" subscription-manager register \
    --activationkey="$ACTIVATION_KEY" \
    --org="$ORG_ID"

echo "Installing packages..."
$CONTAINER_RUNTIME exec "$CONTAINER_NAME" dnf install -y skopeo git make python3-pip

if [[ -n "$REGISTRY_USERNAME" && -n "$REGISTRY_PASSWORD" ]]; then
    echo "Logging into registry with skopeo..."
    REGISTRY_HOST=$(echo "$BASE_IMAGE" | cut -d'/' -f1)
    $CONTAINER_RUNTIME exec "$CONTAINER_NAME" skopeo login "$REGISTRY_HOST" \
        --username "$REGISTRY_USERNAME" \
        --password "$REGISTRY_PASSWORD"
fi

echo "Installing rpm-lockfile-prototype..."
$CONTAINER_RUNTIME exec "$CONTAINER_NAME" python3 -m pip install --user \
    https://github.com/konflux-ci/rpm-lockfile-prototype/archive/refs/tags/v0.21.0.tar.gz

echo "Creating workdir and copying files..."
$CONTAINER_RUNTIME exec "$CONTAINER_NAME" mkdir -p "$WORKDIR"
$CONTAINER_RUNTIME cp "$INPUT_FILE" "$CONTAINER_NAME:$WORKDIR/$(basename "$INPUT_FILE")"
$CONTAINER_RUNTIME cp "$REDHAT_REPO_FILE" "$CONTAINER_NAME:$WORKDIR/$(basename "$REDHAT_REPO_FILE")"

echo "Running rpm-lockfile-prototype..."
$CONTAINER_RUNTIME exec -w "$WORKDIR" "$CONTAINER_NAME" bash -c '
    DNF_VAR_SSL_CLIENT_KEY=$(find /etc/pki/entitlement -type f -name "*key.pem" | head -1)
    export DNF_VAR_SSL_CLIENT_KEY
    DNF_VAR_SSL_CLIENT_CERT="${DNF_VAR_SSL_CLIENT_KEY//-key/}"
    export DNF_VAR_SSL_CLIENT_CERT
    /root/.local/bin/rpm-lockfile-prototype \
        --image "'"$BASE_IMAGE"'" \
        --outfile "'"$OUTPUT_FILE"'" \
        "'"$(basename "$INPUT_FILE")"'"
'

echo "Copying output file from container..."
$CONTAINER_RUNTIME cp "$CONTAINER_NAME:$WORKDIR/$OUTPUT_FILE" "$OUTPUT_FILE"

echo "Successfully generated $OUTPUT_FILE"
