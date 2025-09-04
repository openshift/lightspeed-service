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
        
    # cleanup the bundle
    operator-sdk cleanup lightspeed-operator
    
    # delete the OLS namespace
    oc delete --wait --ignore-not-found ns openshift-lightspeed

    # delete the ImageDigestMirrorSet
    oc delete --wait --ignore-not-found imagedigestmirrorset/openshift-lightspeed-prod-to-ci
}

# $1 suite id
# $2 which test tags to include
# $3 PROVIDER
# $4 PROVIDER_KEY_PATH
# $5 MODEL
# $6 OLS_IMAGE
# $7 OLS_CONFIG_SUFFIX
function run_suite() {
  echo "Preparing to run suite $1"
  
  if [ "$1" = "model_eval" ]; then
  # Run eval tests
    SUITE_ID=$1 TEST_TAGS=$2 PROVIDER=$3 PROVIDER_KEY_PATH=$4 MODEL=$5 OLS_IMAGE=$6 OLS_CONFIG_SUFFIX=$7 ARTIFACT_DIR=$ARTIFACT_DIR make test-eval
  else
  # Run e2e tests
    SUITE_ID=$1 TEST_TAGS=$2 PROVIDER=$3 PROVIDER_KEY_PATH=$4 MODEL=$5 OLS_IMAGE=$6 OLS_CONFIG_SUFFIX=$7 ARTIFACT_DIR=$ARTIFACT_DIR make test-e2e
  fi
  local rc=$?
  return $rc
}
