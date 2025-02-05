#!/usr/bin/env bash

set -o nounset
set -o pipefail
set -x

CI_SERVER_URL=https://prow.svc.ci.openshift.org/view/gcs/origin-ci-test
COVER_PROFILE=${COVER_PROFILE:-"$1"}
JOB_TYPE=${JOB_TYPE:-"local"}

# Configure the git refs and job link based on how the job was triggered via prow
if [[ "${JOB_TYPE}" == "presubmit" ]]; then
       echo "detected PR code coverage job for #${PULL_NUMBER}"
       REF_FLAGS="-P ${PULL_NUMBER} -C ${PULL_PULL_SHA}"
       JOB_LINK="${CI_SERVER_URL}/pr-logs/pull/${REPO_OWNER}_${REPO_NAME}/${PULL_NUMBER}/${JOB_NAME}/${BUILD_ID}"
elif [[ "${JOB_TYPE}" == "batch" ]] || [[ "${JOB_TYPE}" == "postsubmit" ]]; then
       echo "detected branch code coverage job for ${PULL_BASE_REF}"
       REF_FLAGS="-B ${PULL_BASE_REF} -C ${PULL_BASE_SHA}"
       JOB_LINK="${CI_SERVER_URL}/logs/${JOB_NAME}/${BUILD_ID}"
elif [[ "${JOB_TYPE}" == "local" ]]; then
       echo "coverage report available at ${COVER_PROFILE}"
       exit 0
else
       echo "${JOB_TYPE} jobs not supported" >&2
       exit 1
fi

# Configure certain internal codecov variables with values from prow.
export CI_BUILD_URL="${JOB_LINK}"
export CI_BUILD_ID="${JOB_NAME}"
export CI_JOB_ID="${BUILD_ID}"

if [[ "${JOB_TYPE}" != "local" ]]; then
       if [[ -z "${ARTIFACT_DIR:-}" ]] || [[ ! -d "${ARTIFACT_DIR}" ]] || [[ ! -w "${ARTIFACT_DIR}" ]]; then
              # shellcheck disable=SC2016
              echo '${ARTIFACT_DIR} must be set for non-local jobs, and must point to a writable directory' >&2
              exit 1
       fi
       curl -sS https://codecov.io/bash -o "${ARTIFACT_DIR}/codecov.sh"
       bash <(cat "${ARTIFACT_DIR}/codecov.sh") -Z -K -f "${COVER_PROFILE}" -r "${REPO_OWNER}/${REPO_NAME}" "${REF_FLAGS}"
       # shellcheck disable=SC2181
       if [ $? -ne 0 ]; then
              echo "Failed uploading coverage report from a non local environment. Exiting gracefully with status code 0."
              exit 0
       fi
else
       bash <(curl -s https://codecov.io/bash) -Z -K -f "${COVER_PROFILE}" -r "${REPO_OWNER}/${REPO_NAME}" "${REF_FLAGS}"
       # shellcheck disable=SC2181
       if [ $? -ne 0 ]; then
              echo "Failed uploading coverage report from local environment. Exiting gracefully with status code 0."
              exit 0
       fi
fi
