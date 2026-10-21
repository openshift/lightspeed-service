# HyperShift Image Mirrors Implementation Plan

**Goal:** Run the service e2e installer on HyperShift without guest-side ImageDigestMirrorSet mutations.

**Architecture:** Pass the existing mirror mappings to the provisioner's supported IMAGE_CONTENT_SOURCES environment input. Set OLS_EXTERNAL_IMAGE_MIRRORS=true in the Konflux test step to suppress installer and cleanup mutations, retaining default behavior on standalone clusters.

**Tech Stack:** Tekton YAML, Python/pytest, Bash.

## Tasks
1. Add failing regressions to `.tekton/integration-tests/test_pipeline.py`, `tests/unit/e2e/test_ols_installer.py`, and `tests/unit/e2e/test_e2e_cluster_scripts.py`. Cover mirror mapping parity, external-management wiring, installer behavior in both modes, and shell cleanup in both modes.
2. Wire provisioning mirrors and the test environment in `.tekton/integration-tests/pipelines/lightspeed-service-e2e-test-pipeline.yaml`. Guard mirror application in `tests/e2e/utils/ols_installer.py` and deletion in `tests/scripts/utils.sh`. Remove duplicate mirror application.
3. Document the opt-in and provisioning contract in `.tekton/integration-tests/README.md`.
4. Run offline pipeline tests, focused pytest tests, Bash syntax validation, and diff checks. A live Konflux run is required to verify cluster provisioning and actual image pulls.

## Alternatives
Skipping mirrors altogether risks unresolved image pulls. Bypassing HyperShift admission policy violates its management model. Provisioning mirrors and explicitly opting out of guest mutations preserves the supported lifecycle.
