#!/bin/bash -e
# Input env variables:
# - PROVIDER - the LLM provider to be used during the test
# - PROVIDER_KEY_PATH - path to a file containing the credentials to be used with the llm provider
# - MODEL - name of the model to use during e2e testing

# Script flow:
# 1) Generate an OLS config file from env variables
# 2) Launch an OLS server via the run Makefile target and waits for it to be ready
# 3) Invoke the test-e2e Makefile target
# 4) Terminate the OLS server

DIR="${BASH_SOURCE%/*}"
if [[ ! -d "$DIR" ]]; then DIR="$PWD"; fi
. "$DIR/utils.sh"

# temp directory for generated resource yamls
TMPDIR=$(mktemp -d)

# ARTIFACT_DIR is defined when running in a prow job, content
# in this location is automatically collected at the end of the test job
ARTIFACT_DIR=${ARTIFACT_DIR:-$TMPDIR}

# Deletes may fail if this is the first time running against
# the cluster, so ignore failures
oc delete ns openshift-lightspeed || true
oc delete clusterrole ols-sar-check || true
oc delete clusterrolebinding ols-sar-check || true
oc delete clusterrole ols-user || true

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
OLS_TOKEN=$(oc create token olsuser)

# grant the service account permission to query ols
oc adm policy add-cluster-role-to-user ols-user -z olsuser

# determine the hostname for the ols route
OLS_URL=https://$(oc get route ols -o jsonpath='{.spec.host}')

# wait for the ols api server to come up
wait_for_ols

make test-e2e

#curl -k -X 'POST' 'https://lightspeed-w-rag-openshift-lightspeed.apps.bparees.devcluster.openshift.com/v1/query' -H 'accept: application/json' -H 'Content-Type: application/json' -H "Authorization: bearer $USER_TOKEN" -d '{"query": "write a deployment yaml for the mongodb image"}'
