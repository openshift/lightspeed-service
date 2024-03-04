#!/bin/bash
set -eou pipefail

# Input env variables:
# - PROVIDER - the LLM provider to be used during the test
# - PROVIDER_KEY_PATH - path to a file containing the credentials to be used with the llm provider
# - PROVIDER_PROJECT_ID - for use with watsonx, can leave empty otherwise
# - MODEL - name of the model to use during e2e testing
# - OLS_IMAGE - pullspec for the ols image to deploy on the cluster


# Script flow:
# 1) Install OLS into a namespace on the cluster with valid config/api tokens and exposed via a route
# 2) Setup a service account w/ permission to access OLS
# 3) Wait for the ols api server to be available
# 4) Invoke the test-e2e Makefile target

make install-deps && make install-deps-test

DIR="${BASH_SOURCE%/*}"
if [[ ! -d "$DIR" ]]; then DIR="$PWD"; fi
. "$DIR/utils.sh"

function finish() {
    if [ "${LOCAL_MODE:-0}" -eq 1 ]; then
      # When running locally, just cleanup the tmp files
      rm -rf "$ARTIFACT_DIR"
    else
      # when running as a ci job, gather more info from the cluster
      # to be included in the job artifacts and don't cleanup
      # the artifact dir so it can be archived w/ the job
      must_gather
    fi
}
trap finish EXIT

# ARTIFACT_DIR is defined when running in a prow job, content
# in this location is automatically collected at the end of the test job
# If ARTIFACT_DIR is not defined, we are running locally on a developer machine
if [ -z "${ARTIFACT_DIR:-}" ]; then
    # temp directory for generated resource yamls
    readonly ARTIFACT_DIR=$(mktemp -d)
    # Clean up the tmpdir on exit
    readonly LOCAL_MODE=1
fi


# Deletes may fail if this is the first time running against
# the cluster, so ignore failures
oc delete --wait --ignore-not-found ns openshift-lightspeed
oc delete --wait --ignore-not-found clusterrole ols-sar-check
oc delete --wait --ignore-not-found clusterrolebinding ols-sar-check
oc delete --wait --ignore-not-found clusterrole ols-user

oc create ns openshift-lightspeed
oc project openshift-lightspeed

# create the llm api key secret ols will mount
oc create secret generic llmcreds --from-file=llmkey="$PROVIDER_KEY_PATH"

# create the configmap containing the ols config yaml
envsubst < tests/config/cluster_install/ols_configmap.yaml > "$ARTIFACT_DIR/ols_configmap.yaml"
oc create -f "$ARTIFACT_DIR/ols_configmap.yaml"

# create the ols deployment and related resources (service, route, rbac roles)
envsubst < tests/config/cluster_install/ols_manifests.yaml > "$ARTIFACT_DIR/ols_manifests.yaml"
oc create -f "$ARTIFACT_DIR/ols_manifests.yaml"

# create a new service account with no special permissions and get an auth token for it
oc create sa olsuser
export OLS_TOKEN=$(oc create token olsuser)

# grant the service account permission to query ols
oc adm policy add-cluster-role-to-user ols-user -z olsuser

# determine the hostname for the ols route
export OLS_URL=https://$(oc get route ols -o jsonpath='{.spec.host}')

# wait for the ols api server to come up
wait_for_ols "$OLS_URL"

make test-e2e
