#!/bin/bash
# Verify that Konflux hermetic build hash files are in sync with uv.lock.
# Catches PRs that update pyproject.toml/uv.lock without updating hash files.
set -euo pipefail

SOURCE_HASH_FILE=".konflux/requirements.hashes.source.txt"
WHEEL_HASH_FILE=".konflux/requirements.hashes.wheel.txt"
WHEEL_PYPI_HASH_FILE=".konflux/requirements.hashes.wheel.pypi.txt"

# Packages in uv.lock runtime that are legitimately absent from hash files.
# Keep this list minimal — a growing allowlist is a red flag.
EXPECTED_MISSING=(
    # Platform-specific (not built for linux x86_64)
    pywin32

    # Test packages pulled in transitively via langchain-google-vertexai
    # → langchain-tests → pytest (and its plugin ecosystem)
    iniconfig
    langchain_tests
    llama_index_cli
    pluggy
    py_cpuinfo
    pytest
    pytest_asyncio
    pytest_benchmark
    pytest_codspeed
    pytest_recording
    pytest_socket
    syrupy
    vcrpy

    # Transitive deps resolved differently by uv export vs uv pip compile
    # (extras, conditional markers, RHOAI overrides)
    durationpy
    grpcio_status
    h2
    hpack
    hyperframe
    importlib_metadata
)

log() { echo "==> $*"; }

# PEP 503 normalization: lowercase, collapse [-_.]+ to single underscore
normalize_names() {
    sed 's/ *==.*//' | tr '[:upper:]' '[:lower:]' | sed 's/[-_.]\{1,\}/_/g' | sort -u
}

# --- Extract runtime packages from uv.lock ---
log "Exporting runtime dependencies from uv.lock"
UV_RUNTIME=$(uv export --locked --no-dev --no-editable --no-header --no-annotate --format requirements.txt \
    | grep -E '^[a-zA-Z0-9]' \
    | normalize_names)
UV_COUNT=$(echo "$UV_RUNTIME" | wc -l | tr -d ' ')

# --- Validate hash files exist ---
for hash_file in "$SOURCE_HASH_FILE" "$WHEEL_HASH_FILE" "$WHEEL_PYPI_HASH_FILE"; do
    if [[ ! -r "$hash_file" ]]; then
        echo "ERROR: required hash file is missing or unreadable: $hash_file" >&2
        exit 1
    fi
done

# --- Extract packages from all three hash files ---
log "Parsing hash files"
HASH_PKGS=$( {
    grep -E '^[a-zA-Z0-9]' "$SOURCE_HASH_FILE"
    grep -E '^[a-zA-Z0-9]' "$WHEEL_HASH_FILE"
    grep -E '^[a-zA-Z0-9]' "$WHEEL_PYPI_HASH_FILE" || true
} | normalize_names)
HASH_COUNT=$(echo "$HASH_PKGS" | grep -c . || true)

# --- Build allowlist set ---
ALLOW_SET=$(printf '%s\n' "${EXPECTED_MISSING[@]}" | normalize_names)
ALLOW_COUNT=$(echo "$ALLOW_SET" | grep -c . || true)

# --- Compute missing = (uv_runtime - hash_pkgs - allowlist) ---
MISSING=$(comm -23 <(echo "$UV_RUNTIME") <(echo "$HASH_PKGS") \
    | comm -23 - <(echo "$ALLOW_SET"))
MISSING_COUNT=$(echo "$MISSING" | grep -c . || true)
COVERED=$((HASH_COUNT + ALLOW_COUNT))

# --- Check for stale allowlist entries ---
STALE=$(
    {
        comm -23 <(echo "$ALLOW_SET") <(echo "$UV_RUNTIME")
        comm -12 <(echo "$ALLOW_SET") <(echo "$HASH_PKGS")
    } | sed '/^$/d' | sort -u
)
STALE_COUNT=$(echo "$STALE" | grep -c . || true)

# --- Report ---
log "$UV_COUNT runtime packages in uv.lock, $ALLOW_COUNT allowlisted, $HASH_COUNT in hash files"

EXIT_CODE=0

if [[ $MISSING_COUNT -gt 0 ]]; then
    echo ""
    echo "ERROR: $MISSING_COUNT package(s) in uv.lock but NOT in any hash file or allowlist:"
    echo "$MISSING" | sed 's/^/  - /'
    echo ""
    echo "Fix: run 'make konflux-requirements' or surgically add the missing packages."
    echo "See CLAUDE.md for the surgical approach."
    EXIT_CODE=1
fi

if [[ $STALE_COUNT -gt 0 ]]; then
    echo ""
    echo "ERROR: $STALE_COUNT allowlist entry/entries no longer in uv.lock (stale):"
    echo "$STALE" | sed 's/^/  - /'
    echo ""
    echo "Fix: remove stale entries from EXPECTED_MISSING in this script."
    EXIT_CODE=1
fi

if [[ $EXIT_CODE -eq 0 ]]; then
    log "All runtime packages accounted for."
fi

exit $EXIT_CODE
