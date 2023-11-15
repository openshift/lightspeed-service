#!/bin/bash

# Version
OLS_VERSION=v0.0.0

# To build container for local use
podman build --no-cache --build-arg=VERSION=${OLS_VERSION} -t ols:latest -f Containerfile

# To publish to a registry tag registry name
# podman tag ols:latest quay.io/<org>/ols:$OLS_VERSION

# To test-run for local development
# podman run --rm -ti -p 8000:8000 localhost/ols:latest