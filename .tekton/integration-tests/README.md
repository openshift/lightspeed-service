# Service e2e tests on HyperShift

The pipeline validates the `lightspeed-service` component in the Konflux `SNAPSHOT`, provisions the same AWS HyperShift workflow used by lightspeed-operator, checks out `.source.git.revision`, and runs `tests/scripts/test-e2e-cluster.sh`. It passes `.containerImage` as `OLS_IMAGE` to the harness's existing deployment image replacement. It does not obtain the test revision from the operator's related-images file.

The harness installs the operator itself. `bundle-image` defaults to `quay.io/openshift-lightspeed/lightspeed-operator-bundle:latest`; pass a digest to pin the operator independently of the service build. `openshift-version-prefix` defaults to `4.22` (no trailing dot), matching the operator pipeline. Additional scenarios can reuse the same pipeline for other supported OpenShift versions.

## Enable

Merge these files before applying the scenario: its Git resolver reads `main` from `openshift/lightspeed-service`. Apply `integration-test-scenarios/konflux-integration-test-scenarios.yaml` to the existing `crt-nshift-lightspeed-tenant` namespace. The scenario is initially optional and uses `component_lightspeed-service`, matching the build PipelineRuns.

The tenant must have access to the `aws-konflux-prod` cluster profile and Git task/StepAction resolvers. The pipeline reuses the secure OCI publication StepAction from `openshift/lightspeed-operator` at `main`; that file must be available there before running this pipeline. Like the operator pipeline, provisioning uses three `m5.2xlarge` workers on `hosted-mgmt2`. Cluster ownership is passed to the provisioner for lifecycle management.

Required secrets (same names as the operator's service integration pipeline):

- `azureopenai-apitoken`, `azureopenai-entra-id`, `bam-apitoken`, `openai`, `watsonx-apitoken`, `vertex-apitoken`
- `bedrock-apitoken`, `bedrock-role-apitoken`
- `insights-stage-upload-offline-token`
- `ols-konflux-artifacts-bot`: requires `oci-storage-dockerconfigjson` for secure artifact publication **and** `.dockerconfigjson` for the catalog log-export task, with push permission to `quay.io/openshift-lightspeed/ols-service-artifacts`

Provisioning supplies the cluster kubeconfig secret. The test step copies it to a writable location before invoking the harness. The test image requires permission to install RPMs and network access for Git, uv dependencies, the operator SDK, and provider APIs.

Test artifacts are scanned and published from `/workspace/artifacts` to a per-run tag `<PipelineRun name>-<service commit>`. The source checkout and virtual environment are not published. Publication continues after a test failure, then `fail-if-any-step-failed` preserves failure status. Pipeline logs are exported in `finally`.

## Offline checks

Requires Python, PyYAML, and jq:

```sh
python .tekton/integration-tests/test_pipeline.py
```

These standalone checks execute the snapshot extraction with valid and invalid snapshots and verify pipeline/scenario wiring. They do not provision infrastructure, validate admission against the tenant's Tekton version, or exercise provider APIs. A real Konflux run is still required to verify installation, image pulls, cluster cleanup, and artifact publication.
