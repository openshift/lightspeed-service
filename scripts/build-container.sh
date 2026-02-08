#!/bin/bash

# Version
OLS_VERSION=v1.0.10

# To build container for local use
if [ -z "$OLS_NO_IMAGE_CACHE" ]; then
  podman build --no-cache --build-arg=VERSION="${OLS_VERSION}" -t "${OLS_API_IMAGE:-quay.io/openshift-lightspeed/lightspeed-service-api:latest}" -f Containerfile
else
  podman build --build-arg=VERSION=${OLS_VERSION} -t "${OLS_API_IMAGE:-quay.io/openshift-lightspeed/lightspeed-service-api:latest}" -f Containerfile
fi

# To test-run for local development
# podman run --rm -ti -p 8080:8080 ${OLS_API_IMAGE:-quay.io/openshift-lightspeed/lightspeed-service-api:latest}
