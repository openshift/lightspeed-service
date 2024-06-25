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

# no arguments
function cleanup_ols() {
    # Deletes may fail if this is the first time running against
    # the cluster, so ignore failures
    oc delete --wait --ignore-not-found ns openshift-lightspeed
    oc delete --wait --ignore-not-found clusterrole ols-sar-check
    oc delete --wait --ignore-not-found clusterrolebinding ols-sar-check
    oc delete --wait --ignore-not-found clusterrole ols-user
}

# no arguments
function cleanup_ols_operator() {
    # Deletes may fail if this is the first time running against
    # the cluster, so ignore failures

    # delete the OLS operand
    oc delete --wait --ignore-not-found olsconfig/cluster

    # delete the OLS subscription
    oc delete -n openshift-lightspeed --wait --ignore-not-found subscription/lightspeed-operator
    oc delete -n openshift-lightspeed --wait --ignore-not-found operatorgroup/openshift-lightspeed

    # delete the OLS catalog
    oc delete -n openshift-marketplace --wait --ignore-not-found catalogsource/lightspeed-operator-catalog

    # delete the OLS namespace
    oc delete --wait --ignore-not-found ns openshift-lightspeed
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

# Arguments
# SuiteID
# Provider
# Provider key path
# Provider url (if needed, azure openai only normally)
# Provider project id (if needed, watsonx only)
# Deployment name (if needed, azure openai only)
# Model
# OLS image
function install_ols_operator() {

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


    # install the operator catalog
    oc create -f tests/config/operator_install/catalog.yaml
    oc create -f tests/config/operator_install/operatorgroup.yaml
    oc create -f tests/config/operator_install/subscription.yaml

    installed=0
    for i in {1..30}; do
      echo "Checking install state... attempt $i of 30"
      oc get clusterserviceversion
      oc get clusterserviceversion | grep Succeeded
      if [[ $? -eq 0 ]]; then
        installed=1
        break
      fi
      sleep 6
    done

    if [[ $installed -eq 0 ]]; then
      echo "Failed to install OLS operator"
      echo "Current operator install state:"
      oc get clusterserviceversion
      exit 1;
    fi



    # csv=$(oc get clusterserviceversion -o name)
    # oc get ${csv} -o yaml > /tmp/csv_original.yaml
    # oc get ${csv} -o yaml > /tmp/csv.yaml

    # sed -i "s#\(lightspeed-service=\)quay.io/openshift-lightspeed/lightspeed-service-api:.*#\1${OLS_IMAGE}#g" /tmp/csv.yaml

    # oc apply -f /tmp/csv.yaml
    # updated=0
    # for i in {1..30}; do
    #   echo "Waiting for OLS operator deployment to update... attempt $i of 30"
    #   oc get deployment/lightspeed-operator-controller-manager -o jsonpath={.spec.template.spec.containers[0].args} | grep ${OLS_IMAGE}
    #   if [[ $? -eq 0 ]]; then
    #     updated=1
    #     break
    #   fi
    #   sleep 6
    # done

    # if [[ $updated -eq 0 ]]; then
    #   echo "Deployment failed to update operator to use built operand image ${OLS_IMAGE}"
    #   exit 1
    # fi


    # create the ols config yaml definition
    mkdir -p "$ARTIFACT_DIR/$SUITE_ID"
    
    # envsubst < tests/config/operator_install/olsconfig.crd.${PROVIDER}.yaml > "$ARTIFACT_DIR/$SUITE_ID/olsconfig.yaml.tmp"
    # # If no provider url is being specified, remove the url field from the config yaml
    # # so we use the default provider url values.
    # if [ -z ${PROVIDER_URL:-} ]; then
    #     grep -v url: "${ARTIFACT_DIR}/$SUITE_ID/olsconfig.yaml.tmp" > "${ARTIFACT_DIR}/$SUITE_ID/olsconfig.crd.${PROVIDER}.yaml"
    #     rm "${ARTIFACT_DIR}/$SUITE_ID/olsconfig.yaml.tmp"
    # else
    #     mv "${ARTIFACT_DIR}/$SUITE_ID/olsconfig.yaml.tmp" "${ARTIFACT_DIR}/$SUITE_ID/olsconfig.crd.${PROVIDER}.yaml"
    # fi

    # create the llm api key secret ols will mount
    oc create secret generic llmcreds --from-file=apitoken="$PROVIDER_KEY_PATH"
    # create the olsconfig operand
    oc create -f tests/config/operator_install/olsconfig.crd.${PROVIDER}.yaml

    created=0
    for i in {1..10}; do
        oc get deployment/lightspeed-app-server
        if [[ $? -eq 0 ]]; then
          created=1
          break
        fi
        sleep 6
    done
    if [[ $created -eq 0 ]]; then
      echo "OLS deployment was not created by operator"
      exit 1
    fi


    # Shutdown the OLS operator so we can directly control the operand
    oc scale deployment/lightspeed-operator-controller-manager --replicas 0

    oc patch deployment lightspeed-app-server --type='json' -p="[{'op': 'replace', 'path': '/spec/template/spec/containers/0/image', 'value':${OLS_IMAGE}}]"
    oc patch deployment lightspeed-app-server --type='json' -p="[{'op': 'replace', 'path': '/spec/template/spec/containers/1/image', 'value':${OLS_IMAGE}}]"

    oc set env deployment/lightspeed-app-server -c lightspeed-service-user-data-collector OLS_USER_DATA_COLLECTION_INTERVAL=10 RUN_WITHOUT_INITIAL_WAIT=true \
        INGRESS_ENV=stage CP_OFFLINE_TOKEN=${CP_OFFLINE_TOKEN} LOG_LEVEL=DEBUG

    oc wait deployment/lightspeed-app-server --for=condition=Available=true --timeout=180s
    
    # create a route so tests can access OLS directly
    oc create -f tests/config/operator_install/route.yaml
    
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

  cleanup_ols_operator
  
  install_ols_operator "$1" "$3" "$4" "$5" "$6" "$7" "$8" "$9"

  # Run e2e tests with response evaluation.
  SUITE_ID=$1 TEST_TAGS=$2 PROVIDER=$3 MODEL=$8 ARTIFACT_DIR=$ARTIFACT_DIR make test-e2e

  local rc=$?
  return $rc
}
