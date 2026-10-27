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

    # scale the operator pod to 1 to assure finalizer will run
    oc scale deployment lightspeed-operator-controller-manager -n openshift-lightspeed --replicas=1

    # delete the OLS operand
    oc delete --wait --ignore-not-found olsconfig/cluster
        
    # cleanup the bundle
    operator-sdk cleanup lightspeed-operator
    
    # delete the OLS namespace
    oc delete --wait --ignore-not-found ns openshift-lightspeed

    # Externally managed mirrors belong to HostedCluster, not this test suite.
    if [[ "${OLS_EXTERNAL_IMAGE_MIRRORS:-false}" != "true" ]]; then
        oc delete --wait --ignore-not-found imagedigestmirrorset/openshift-lightspeed-prod-to-ci
    fi
}

# $1 suite id (single mode) or matrix label (matrix mode)
# $2 which test tags to include
# $3 PROVIDER, or "suite:provider:model" entries in matrix mode
# $4 PROVIDER_KEY_PATH, or key paths in matrix entry order in matrix mode
# $5 MODEL
# $6 OLS_IMAGE
# $7 OLS_CONFIG_SUFFIX
function run_suite() {
  echo "Preparing to run suite $1"
  local saved_seconds=$SECONDS

  if [[ "$1" == lseval_periodic* ]]; then
  # Run LSEval periodic tests (full 797-question dataset)
    SUITE_ID=$1 TEST_TAGS=$2 PROVIDER=$3 PROVIDER_KEY_PATH=$4 MODEL=$5 OLS_IMAGE=$6 OLS_CONFIG_SUFFIX=$7 ARTIFACT_DIR=$ARTIFACT_DIR make test-lseval-periodic
  elif [[ "$1" == lseval_presubmit* || "$1" == lseval_daily* ]]; then
  # Run LSEval short-dataset tests (presubmit or daily)
    SUITE_ID=$1 TEST_TAGS=$2 PROVIDER=$3 PROVIDER_KEY_PATH=$4 MODEL=$5 OLS_IMAGE=$6 OLS_CONFIG_SUFFIX=$7 ARTIFACT_DIR=$ARTIFACT_DIR make test-lseval-presubmit
  elif [ "$1" = "cluster_updates" ]; then
  # Run cluster-updates evaluation tests (18 conversations, 35 evaluations)
    SUITE_ID=$1 TEST_TAGS=$2 PROVIDER=$3 PROVIDER_KEY_PATH=$4 MODEL=$5 OLS_IMAGE=$6 OLS_CONFIG_SUFFIX=$7 ARTIFACT_DIR=$ARTIFACT_DIR make test-cluster-updates
  elif [[ "$3" == *":"* ]]; then
  # Provider matrix mode: every "suite:provider:model" entry runs against one
  # deployment. The first entry applies the OLSConfig for all providers (its
  # provider must match the CR default so no switch restart is needed); later
  # entries set SKIP_CLUSTER_SETUP and only switch default_provider/model,
  # which patches the configmap and restarts the app-server.
  # E2E_MATRIX_CREDENTIAL_PROVIDERS/E2E_MATRIX_CREDENTIAL_KEY_PATHS override
  # the bootstrap lists for the first iteration: credentials must exist for
  # every provider configured in the combined CR, including providers whose
  # tests run only in the periodic job — the CR cannot reconcile when a
  # configured provider lacks its secret.
    local matrix_rc=0
    local entries keypaths
    read -r -a entries <<< "$3"
    read -r -a keypaths <<< "$4"
    if [ "${#entries[@]}" -ne "${#keypaths[@]}" ]; then
      echo "ERROR: suite $1 has ${#entries[@]} matrix entries but ${#keypaths[@]} key paths"
      return 2
    fi
    local all_providers=""
    local index=0
    local entry suite_id entry_provider entry_model keypath
    for entry in "${entries[@]}"; do
      IFS=':' read -r _ entry_provider _ <<< "$entry"
      all_providers="$all_providers $entry_provider"
    done
    all_providers=${all_providers# }
    # E2E_MATRIX_FILTER: space-separated suite ids to rerun a subset of the
    # matrix, e.g. E2E_MATRIX_FILTER="azure_openai_tool_calling" after a
    # single provider failed. The first selected entry performs the full
    # setup (CR apply with every provider) and then switches to its provider,
    # so any subset is valid. A matrix whose entries are all filtered out is
    # skipped (the filter may target the other tool-calling matrix); the
    # filter is validated against the union of all matrices at the end of
    # run_suites via check_matrix_filter_matches. The filter narrows only
    # the tool-calling matrices — all other suites still run.
    if [ -n "${E2E_MATRIX_FILTER:-}" ]; then
      local filtered_entries=()
      local filtered_keypaths=()
      index=0
      for entry in "${entries[@]}"; do
        IFS=':' read -r suite_id _ _ <<< "$entry"
        if [[ " $E2E_MATRIX_FILTER " == *" $suite_id "* ]]; then
          filtered_entries+=("$entry")
          filtered_keypaths+=("${keypaths[$index]}")
        fi
        index=$((index + 1))
      done
      if [ "${#filtered_entries[@]}" -eq 0 ]; then
        echo "E2E_MATRIX_FILTER='$E2E_MATRIX_FILTER' matched no entries of suite $1 - skipping" >&2
        return 0
      fi
      echo "E2E_MATRIX_FILTER active: running ${#filtered_entries[@]} of ${#entries[@]} entries for suite $1"
      printf '%s\n' "${filtered_entries[@]%%:*}" >> "${ARTIFACT_DIR}/.matrix_filter_matched" 2>/dev/null || true
      entries=("${filtered_entries[@]}")
      keypaths=("${filtered_keypaths[@]}")
    fi
    index=0
    for entry in "${entries[@]}"; do
      local iteration_start=$SECONDS
      local skip_setup=false
      IFS=':' read -r suite_id entry_provider entry_model <<< "$entry"
      keypath=${keypaths[$index]}
      index=$((index + 1))
      if [ $index -eq 1 ]; then
        # the first iteration creates every provider's credentials and
        # applies the combined OLSConfig. Credentials must cover every
        # provider configured in the CR (E2E_MATRIX_CREDENTIAL_* override the
        # tested entries for exactly that); the key paths pair with the
        # bootstrap provider list.
        SUITE_ID=$suite_id TEST_TAGS=$2 PROVIDER=${E2E_MATRIX_CREDENTIAL_PROVIDERS:-$all_providers} PROVIDER_KEY_PATH=${E2E_MATRIX_CREDENTIAL_KEY_PATHS:-$4} MODEL=$entry_model OLS_IMAGE=$6 OLS_CONFIG_SUFFIX=$7 ARTIFACT_DIR=$ARTIFACT_DIR SKIP_CLUSTER_SETUP=$skip_setup MUST_GATHER_ON_SUCCESS=false SWITCH_DEFAULT_PROVIDER="$entry_provider $entry_model" make test-e2e
      else
        skip_setup=true
        SUITE_ID=$suite_id TEST_TAGS=$2 PROVIDER=$entry_provider PROVIDER_KEY_PATH=$keypath MODEL=$entry_model OLS_IMAGE=$6 OLS_CONFIG_SUFFIX=$7 ARTIFACT_DIR=$ARTIFACT_DIR SKIP_CLUSTER_SETUP=$skip_setup MUST_GATHER_ON_SUCCESS=false SWITCH_DEFAULT_PROVIDER="$entry_provider $entry_model" make test-e2e
      fi
      local iteration_rc=$?
      if [ $iteration_rc -ne 0 ] && [ $matrix_rc -eq 0 ]; then
        matrix_rc=$iteration_rc
      fi
      local iteration_duration=$(( SECONDS - iteration_start ))
      echo "{\"suite\":\"$suite_id\",\"rc\":$iteration_rc,\"duration_s\":$iteration_duration,\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "${ARTIFACT_DIR}/suite_results.jsonl"
    done
    local duration_s=$(( SECONDS - saved_seconds ))
    echo "Suite $1 matrix finished in ${duration_s}s with rc=$matrix_rc"
    return $matrix_rc
  else
  # Run e2e tests
    local gather_on_success=false
    if [ "$1" = "tool_result_inspection" ]; then
      # the per-provider artifact split needs the gathered podlogs
      gather_on_success=true
    fi
    SUITE_ID=$1 TEST_TAGS=$2 PROVIDER=$3 PROVIDER_KEY_PATH=$4 MODEL=$5 OLS_IMAGE=$6 OLS_CONFIG_SUFFIX=$7 ARTIFACT_DIR=$ARTIFACT_DIR MUST_GATHER_ON_SUCCESS=$gather_on_success make test-e2e
  fi
  local rc=$?
  # Restore per-provider JUnit reports and app-server log slices for the
  # consolidated inspection suite (single deployment, per-request provider).
  if [ "$1" = "tool_result_inspection" ]; then
    if ! uv run python tests/scripts/split_inspection_artifacts.py "$ARTIFACT_DIR" "$1"; then
      echo "WARNING: splitting inspection artifacts for suite $1 failed"
    fi
  fi
  local duration_s=$(( SECONDS - saved_seconds ))
  echo "{\"suite\":\"$1\",\"rc\":$rc,\"duration_s\":$duration_s,\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "${ARTIFACT_DIR}/suite_results.jsonl"
  return $rc
}

# Validate E2E_MATRIX_FILTER against the union of all provider matrices that
# ran in this job. Call once at the end of run_suites: a matrix with no
# selected entries skips silently by design (the filter may target the other
# tool-calling matrix), so an ID that matched nowhere can only be detected
# after every matrix had its chance.
function check_matrix_filter_matches() {
  if [ -z "${E2E_MATRIX_FILTER:-}" ]; then
    return 0
  fi
  local matches
  matches=$(cat "${ARTIFACT_DIR}/.matrix_filter_matched" 2>/dev/null || true)
  if [ -z "$matches" ]; then
    echo "ERROR: E2E_MATRIX_FILTER='$E2E_MATRIX_FILTER' matched no matrix entry in any suite" >&2
    return 1
  fi
  echo "E2E_MATRIX_FILTER matched: $(echo $matches | tr '\n' ' ')"
}
