#!/bin/bash
set -euo pipefail

SCAN_DIR="/tmp/ols-tls-scan"
OLS_PORT=8443
SCANNER_IMAGE="${SCANNER_IMAGE:-quay.io/openshift-lightspeed/ols-qe:tls-scanner}"
OLS_PID=""
FAILURES=0

log() {
    echo "[tls-scan] $*"
}

cleanup() {
    if [[ -n "${OLS_PID}" ]]; then
        kill "${OLS_PID}" 2>/dev/null || true
        wait "${OLS_PID}" 2>/dev/null || true
    fi
    podman rm "tls-scanner-extract-$$" 2>/dev/null || true
    if [[ "${TLS_SCAN_KEEP_ARTIFACTS:-0}" != "1" ]]; then
        rm -rf "${SCAN_DIR}"
        log "Cleaned up ${SCAN_DIR}"
    else
        log "Artifacts preserved in ${SCAN_DIR}"
    fi
}

trap cleanup EXIT

generate_certs() {
    log "Generating TLS certificates..."
    mkdir -p "${SCAN_DIR}"
    openssl req -x509 -newkey rsa:4096 -sha256 -days 1 -noenc \
        -keyout "${SCAN_DIR}/server.key" \
        -out "${SCAN_DIR}/server.crt" \
        -subj "/CN=localhost" \
        -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1" \
        2>/dev/null
    log "Certificates generated in ${SCAN_DIR}"
}

extract_scanner() {
    log "Extracting tls-scanner from ${SCANNER_IMAGE}..."
    local container_name="tls-scanner-extract-$$"
    podman create --name "${container_name}" "${SCANNER_IMAGE}" >/dev/null 2>&1
    podman cp "${container_name}:/usr/local/bin/tls-scanner" "${SCAN_DIR}/tls-scanner"
    podman cp "${container_name}:/opt/testssl" "${SCAN_DIR}/testssl"
    podman rm "${container_name}" >/dev/null 2>&1
    chmod +x "${SCAN_DIR}/tls-scanner"
    export PATH="${SCAN_DIR}/testssl:${PATH}"
    log "tls-scanner extracted"
}

generate_config() {
    local profile_type="$1"
    local config_file="${SCAN_DIR}/config-${profile_type}.yaml"
    log "Generating OLS config for ${profile_type}..."
    cat > "${config_file}" <<EOF
llm_providers:
  - name: test-provider
    type: openai
    url: "http://localhost:1234"
    credentials_path: tests/config/secret/apitoken
    models:
      - name: test-model

ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 10
  default_provider: test-provider
  default_model: test-model
  logging_config:
    app_log_level: warning
    lib_log_level: warning
  tls_config:
    tls_certificate_path: ${SCAN_DIR}/server.crt
    tls_key_path: ${SCAN_DIR}/server.key
  tlsSecurityProfile:
    type: ${profile_type}

dev_config:
  disable_auth: true
  disable_tls: false
EOF
    log "Config written to ${config_file}"
}

start_ols() {
    local profile_type="$1"
    local config_file="${SCAN_DIR}/config-${profile_type}.yaml"
    log "Starting OLS with ${profile_type} profile..."
    OLS_CONFIG_FILE="${config_file}" .venv/bin/python runner.py > "${SCAN_DIR}/ols-${profile_type}.log" 2>&1 &
    OLS_PID=$!
}

wait_for_ols() {
    local max_wait=60
    local waited=0
    log "Waiting for OLS to be ready on port ${OLS_PORT}..."
    while ! curl -sk "https://localhost:${OLS_PORT}/liveness" >/dev/null 2>&1; do
        if ! kill -0 "${OLS_PID}" 2>/dev/null; then
            log "ERROR: OLS process died during startup"
            log "OLS log output:"
            cat "${SCAN_DIR}/ols-"*.log 2>/dev/null || true
            return 1
        fi
        if [[ ${waited} -ge ${max_wait} ]]; then
            log "ERROR: OLS did not become ready within ${max_wait}s"
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done
    log "OLS ready on port ${OLS_PORT} (took ${waited}s)"
}

stop_ols() {
    if [[ -n "${OLS_PID}" ]]; then
        log "Stopping OLS (PID ${OLS_PID})..."
        kill "${OLS_PID}" 2>/dev/null || true
        wait "${OLS_PID}" 2>/dev/null || true
        OLS_PID=""
    fi
}

run_scan() {
    local profile_type="$1"
    log "Running tls-scanner for ${profile_type}..."
    if "${SCAN_DIR}/tls-scanner" \
        -host localhost \
        -port "${OLS_PORT}" \
        -json-file "${SCAN_DIR}/${profile_type}.json" \
        -junit-file "${SCAN_DIR}/${profile_type}-junit.xml" \
        -log-file "${SCAN_DIR}/${profile_type}-scan.log"; then
        log "PASS: ${profile_type}"
    else
        log "FAIL: ${profile_type} (tls-scanner exit code: $?)"
        FAILURES=$((FAILURES + 1))
    fi
}

run_pqc_check() {
    log "Running tls-scanner PQC check (ML-KEM)..."
    if "${SCAN_DIR}/tls-scanner" \
        -host localhost \
        -port "${OLS_PORT}" \
        -pqc-check \
        > "${SCAN_DIR}/pqc-check.log" 2>&1; then
        log "PASS: ML-KEM (post-quantum)"
    else
        log "FAIL: ML-KEM (post-quantum) — tls-scanner exit code: $?"
        log "PQC check output:"
        cat "${SCAN_DIR}/pqc-check.log"
        FAILURES=$((FAILURES + 1))
    fi
}

report_results() {
    local total=3
    local passed=$((total - FAILURES))
    echo ""
    log "=== Results: ${passed}/${total} passed ==="
    if [[ ${FAILURES} -gt 0 ]]; then
        log "FAILED — ${FAILURES} scan(s) did not pass"
        log "Inspect artifacts in ${SCAN_DIR} (re-run with TLS_SCAN_KEEP_ARTIFACTS=1)"
        exit 1
    fi
    log "All TLS scans passed"
}

# --- Main ---
generate_certs
extract_scanner

PASS_NUM=0
for profile in IntermediateType ModernType; do
    PASS_NUM=$((PASS_NUM + 1))
    log "=== Pass ${PASS_NUM}/3: ${profile} ==="
    generate_config "${profile}"
    start_ols "${profile}"
    if wait_for_ols; then
        run_scan "${profile}"
    else
        log "FAIL: ${profile} (OLS did not start)"
        FAILURES=$((FAILURES + 1))
    fi
    stop_ols
done

log "=== Pass 3/3: ML-KEM (post-quantum) ==="
start_ols "ModernType"
if wait_for_ols; then
    run_pqc_check
else
    log "FAIL: ML-KEM (OLS did not start)"
    FAILURES=$((FAILURES + 1))
fi
stop_ols

report_results
