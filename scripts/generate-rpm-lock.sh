#!/bin/bash

# Default values for command-line arguments
BASE_IMAGE="registry.redhat.io/rhai/base-image-cpu-rhel9:3.2"
INPUT_FILE="rpms.in.yaml"
OUTPUT_FILE="rpms.lock.yaml"

usage() {
    echo "Usage: $0 [-i BASE_IMAGE] [-f INPUT_FILE] [-o OUTPUT_FILE]"
    echo ""
    echo "Options:"
    echo "  -i BASE_IMAGE   Base container image (default: $BASE_IMAGE)"
    echo "  -f INPUT_FILE   Input RPM specification file (default: $INPUT_FILE)"
    echo "  -o OUTPUT_FILE  Output lock file (default: $OUTPUT_FILE)"
    echo "  -h              Show this help message"
    exit 1
}


while getopts "i:f:o:h" opt; do
    case $opt in
        i) BASE_IMAGE="$OPTARG" ;;
        f) INPUT_FILE="$OPTARG" ;;
        o) OUTPUT_FILE="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

echo "Using BASE_IMAGE: $BASE_IMAGE"
echo "Using INPUT_FILE: $INPUT_FILE"
echo "Using OUTPUT_FILE: $OUTPUT_FILE"

# check subscription status
if ! sudo subscription-manager status; then
    echo "Failed to check subscription status, please register the system to Red Hat by using the following command:"
    echo "subscription-manager register --org=ORG ID  --activationkey="AK1,AK2,AK3""
    echo "and then run the script again"
    exit 1
fi
echo "Subscription status is OK"

# find the entitlement certificate and key
DNF_VAR_SSL_CLIENT_KEY=$(find /etc/pki/entitlement -type f -name "*key.pem" | head -1)
export DNF_VAR_SSL_CLIENT_KEY
DNF_VAR_SSL_CLIENT_CERT="${DNF_VAR_SSL_CLIENT_KEY//-key/}"
export DNF_VAR_SSL_CLIENT_CERT

rpm-lockfile-prototype --image "$BASE_IMAGE" --outfile "$OUTPUT_FILE" "$INPUT_FILE"
