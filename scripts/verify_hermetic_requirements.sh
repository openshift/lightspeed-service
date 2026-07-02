#!/bin/bash

set -euo pipefail

SOURCE_HASH_FILE="requirements.hashes.source.txt"
WHEEL_HASH_FILE="requirements.hashes.wheel.txt"

# Packages legitimately absent from hermetic build hash files.
# CUDA/nvidia: hermetic build uses CPU-only torch from RHOAI index.
# pywin32: Windows-only.
# typer, shellingham, narwhals, pyopenssl: transitive deps resolved by
# uv export but excluded by konflux_requirements.sh (uv pip compile
# from pyproject.toml resolves a slightly different tree).
EXPECTED_MISSING=(
    cuda_bindings
    cuda_pathfinder
    nvidia_cublas_cu12
    nvidia_cuda_cupti_cu12
    nvidia_cuda_nvrtc_cu12
    nvidia_cuda_runtime_cu12
    nvidia_cudnn_cu12
    nvidia_cufft_cu12
    nvidia_cufile_cu12
    nvidia_curand_cu12
    nvidia_cusolver_cu12
    nvidia_cusparse_cu12
    nvidia_cusparselt_cu12
    nvidia_nccl_cu12
    nvidia_nvjitlink_cu12
    nvidia_nvshmem_cu12
    nvidia_nvtx_cu12
    pywin32
    typer
    shellingham
    narwhals
    pyopenssl
)

TEMP_DIR=$(mktemp -d)

cleanup() {
    rm -rf "$TEMP_DIR"
}

trap cleanup EXIT

log() {
    echo "[verify-hermetic] $*"
}

normalize() {
    tr '[:upper:]' '[:lower:]' | tr '-' '_'
}

log "Exporting runtime dependencies from uv.lock..."
uv export --no-dev --no-editable --format requirements.txt --no-header --no-annotate 2>/dev/null \
    | grep "==" \
    | sed 's/ *==.*//; s/ *\\//' \
    | normalize \
    | sort -u > "$TEMP_DIR/uv_runtime.txt"

uv_count=$(wc -l < "$TEMP_DIR/uv_runtime.txt")

log "Parsing hermetic build hash files..."
grep -hE '^[a-zA-Z]' "$SOURCE_HASH_FILE" "$WHEEL_HASH_FILE" \
    | sed 's/ *==.*//; s/ *\\//' \
    | normalize \
    | sort -u > "$TEMP_DIR/hash_packages.txt"

printf '%s\n' "${EXPECTED_MISSING[@]}" | sort -u > "$TEMP_DIR/expected_missing.txt"
allowlisted=$(comm -12 "$TEMP_DIR/uv_runtime.txt" "$TEMP_DIR/expected_missing.txt" | wc -l)
need_check=$((uv_count - allowlisted))
log "$uv_count runtime packages in uv.lock, $allowlisted allowlisted (CUDA/platform-specific), $need_check must be in hash files"

# Check for stale allowlist entries — packages no longer in uv.lock
comm -23 "$TEMP_DIR/expected_missing.txt" "$TEMP_DIR/uv_runtime.txt" > "$TEMP_DIR/stale.txt"
stale_count=$(wc -l < "$TEMP_DIR/stale.txt")
if [ "$stale_count" -gt 0 ]; then
    log "ERROR: $stale_count stale allowlist entry/entries no longer in uv.lock — remove from EXPECTED_MISSING:"
    while IFS= read -r pkg; do
        echo "  - $pkg"
    done < "$TEMP_DIR/stale.txt"
    exit 1
fi

# missing = uv_runtime - hash_packages - expected_missing
comm -23 "$TEMP_DIR/uv_runtime.txt" "$TEMP_DIR/hash_packages.txt" \
    | comm -23 - "$TEMP_DIR/expected_missing.txt" \
    > "$TEMP_DIR/missing.txt"

missing_count=$(wc -l < "$TEMP_DIR/missing.txt")

if [ "$missing_count" -gt 0 ]; then
    log "ERROR: $missing_count runtime package(s) missing from hermetic build hash files:"
    while IFS= read -r pkg; do
        echo "  - $pkg"
    done < "$TEMP_DIR/missing.txt"
    log "Add these to $SOURCE_HASH_FILE or $WHEEL_HASH_FILE."
    log "See AGENTS.md for the surgical hash file update process."
    exit 1
fi

log "OK — all runtime dependencies are present in hermetic build hash files"
