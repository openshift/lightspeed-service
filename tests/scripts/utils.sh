JUNIT_SETUP_TEMPLATE=$(cat << EOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
    <testsuite name="SUITE_ID">
        <testcase name="setup" classname="SUITE_ID.setup">
            <failure message="OLS failed to start up for SUITE_ID"/>
        </testcase>
    </testsuite>
</testsuites>
EOF
)


# Wait for the ols api server to respond 200 on the readiness endpoint
# $1 - url of the ols server to poll
function wait_for_ols() {
  for i in {1..60}; do
    echo Checking OLS readiness, attempt "$i" of 60
    curl -sk --fail "$1/readiness"
    if [ $? -eq 0 ]; then
      return 0
    fi  
    sleep 6
  done
  return 1
}

# no arguments
function cleanup_ols() {
    # Deletes may fail if this is the first time running against
    # the cluster, so ignore failures
    oc delete --wait --ignore-not-found ns openshift-lightspeed
    oc delete --wait --ignore-not-found clusterrole ols-sar-check
    oc delete --wait --ignore-not-found clusterrolebinding ols-sar-check
    oc delete --wait --ignore-not-found clusterrole ols-user
}

# Arguments
# SuiteID
# Provider
# Provider key path
# Provider url (if needed, azure openai only normally)
# Provider project id (if needed, watsonx only)
# Deployment name (if needed, azure openai only)
# Model
# OLS image
function install_ols() {

    SUITE_ID=$1
    # exports needed for values used by envsubst
    export PROVIDER=$2
    export PROVIDER_KEY_PATH=$3
    export PROVIDER_URL=$4
    export PROVIDER_PROJECT_ID=$5
    export PROVIDER_DEPLOYMENT_NAME=$6
    export MODEL=$7
    export OLS_IMAGE=$8

    # get and export cluster version
    local lines
    mapfile -t lines <<< "$(oc version | grep Server | awk '{print $3}' | awk 'BEGIN{FS="."}{printf("%s\n%s\n", $1, $2)}')"
    export CLUSTER_VERSION_MAJOR="${lines[0]}"
    export CLUSTER_VERSION_MINOR="${lines[1]}"

    oc create ns openshift-lightspeed
    oc project openshift-lightspeed

    # create the llm api key secret ols will mount
    oc create secret generic llmcreds --from-file=llmkey="$PROVIDER_KEY_PATH"

    # create the configmap containing the ols config yaml
    mkdir -p "$ARTIFACT_DIR/$SUITE_ID"
    envsubst < tests/config/cluster_install/ols_configmap.yaml > "$ARTIFACT_DIR/$SUITE_ID/ols_configmap.yaml.tmp"
    # If no provider url is being specified, remove the url field from the config yaml
    # so we use the default provider url values.
    if [ -z ${PROVIDER_URL:-} ]; then
        grep -v url: "${ARTIFACT_DIR}/$SUITE_ID/ols_configmap.yaml.tmp" > "${ARTIFACT_DIR}/$SUITE_ID/ols_configmap.yaml"
        rm "${ARTIFACT_DIR}/$SUITE_ID/ols_configmap.yaml.tmp"
    else
        mv "${ARTIFACT_DIR}/$SUITE_ID/ols_configmap.yaml.tmp" "${ARTIFACT_DIR}/$SUITE_ID/ols_configmap.yaml"
    fi
    oc create -f "$ARTIFACT_DIR/$SUITE_ID/ols_configmap.yaml"

    # create the ols deployment and related resources (service, route, rbac roles)
    envsubst < tests/config/cluster_install/ols_manifests.yaml > "$ARTIFACT_DIR/$SUITE_ID/ols_manifests.yaml"
    oc create -f "$ARTIFACT_DIR/$SUITE_ID/ols_manifests.yaml"

    # determine the hostname for the ols route
    export OLS_URL=https://$(oc get route ols -o jsonpath='{.spec.host}')

}

# $1 suite id
# $2 which test tags to include
# $3 PROVIDER
# $4 PROVIDER_KEY_PATH
# $5 PROVIDER_URL
# $6 PROVIDER_PROJECT_ID
# $7 PROVIDER_DEPLOYMENT_NAME
# $8 MODEL
# $9 OLS_IMAGE
function run_suite() {
  echo "Preparing to run suite $1"

  cleanup_ols
  
  install_ols "$1" "$3" "$4" "$5" "$6" "$7" "$8" "$9"

  # wait for the ols api server to come up
  wait_for_ols "$OLS_URL"
  if [ $? -ne 0 ]; then
    echo "Timed out waiting for OLS to become available"
    echo "${JUNIT_SETUP_TEMPLATE}" | sed "s/SUITE_ID/${1}/g" > $ARTIFACT_DIR/junit_setup_${1}.xml

    ARTIFACT_DIR=$ARTIFACT_DIR SUITE_ID=$1 python tests/scripts/must_gather.py
    return 1
  fi

  # Run e2e tests with response evaluation.
  SUITE_ID=$1 TEST_TAGS=$2 PROVIDER=$3 MODEL=$8 ARTIFACT_DIR=$ARTIFACT_DIR make test-e2e

  local rc=$?
  return $rc
}
