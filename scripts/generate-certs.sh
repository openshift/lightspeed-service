#!/bin/bash

# Define variables
CERT_DIR="./certs"
PRIVATE_KEY="$CERT_DIR/private.key"
CERTIFICATE="$CERT_DIR/certificate.crt"
DAYS_VALID=365
PASSWORD=$1 # Get password from script's first argument

# Create directory for certificates if it doesn't exist
mkdir -p "$CERT_DIR"

if [ -z "$PASSWORD" ]; then
    PASSWORD_PARAM=(-noenc)
else
    PASSWORD_PARAM=(-passout "pass:$PASSWORD")
fi

openssl req -x509 -newkey rsa:4096 -sha256 -days "$DAYS_VALID" "${PASSWORD_PARAM[@]}" \
    -keyout "$PRIVATE_KEY" -out "$CERTIFICATE" -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1"

echo "Certificate and private key have been generated in $CERT_DIR"
