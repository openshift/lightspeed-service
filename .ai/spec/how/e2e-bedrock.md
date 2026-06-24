# E2E Test Infrastructure — AWS Bedrock Provider

Spec for [OLS-3327](https://redhat.atlassian.net/browse/OLS-3327). Adds e2e test coverage for the AWS Bedrock provider, covering both model-prefix routing branches (Anthropic, OpenAI) and both IAM authentication modes (direct, assume-role).

## Dependencies

| Dependency | Status | Gate |
|---|---|---|
| OLS-1895 — service Bedrock provider | Closed | Merged |
| OLS-2605 — operator CRD `bedrock` enum | In Review ([PR #1749](https://github.com/openshift/lightspeed-operator/pull/1749)) | Must merge before e2e runs on cluster |

## Scope

In scope:
- OLSConfig CR templates for Bedrock (Anthropic model, OpenAI model, OpenAI tool calling)
- IAM credential handling in `ols_installer.py` (new `ensure_bedrock_iam_secret()`)
- `create_secrets()` Bedrock branch
- `run_suite` entries in `test-e2e-cluster-periodics.sh` and `test-e2e-cluster.sh`
- Vault credential mapping update in `openshift/release` (separate PR)

Out of scope:
- Bedrock-specific e2e test cases — the existing generic suite runs with `PROVIDER=bedrock`
- Operator CRD changes — covered by OLS-2605

## Credential Architecture

### Two IAM Users

| User | Permission Model | Secret Name |
|---|---|---|
| Direct IAM | IAM policy grants direct Bedrock `InvokeModel` access | `llmcreds` |
| Role-based IAM | IAM user can only `sts:AssumeRole`; the role has Bedrock access | `llmcreds` |

Both secrets are created under the name `llmcreds` (the name referenced by all CR templates). `create_secrets()` deletes and recreates `llmcreds` before each suite.

### Secret Structures

Direct IAM:
```yaml
data:
  aws_access_key_id: <base64>
  aws_secret_access_key: <base64>
```

Role-based IAM:
```yaml
data:
  aws_access_key_id: <base64>
  aws_secret_access_key: <base64>
  role_arn: <base64>
```

### Environment Variables (from Vault)

| Env Var | Purpose |
|---|---|
| `BEDROCK_AWS_ACCESS_KEY_ID` | Direct IAM access key |
| `BEDROCK_AWS_SECRET_ACCESS_KEY` | Direct IAM secret key |
| `BEDROCK_ROLE_AWS_ACCESS_KEY_ID` | Role-based IAM access key |
| `BEDROCK_ROLE_AWS_SECRET_ACCESS_KEY` | Role-based IAM secret key |
| `BEDROCK_ROLE_ARN` | ARN of the role to assume |

### Credential Flow

1. Vault stores credentials for both IAM users.
2. The existing periodic CI job in `openshift/release` maps the five Vault entries to env vars.
3. `create_secrets()` detects `provider_name == "bedrock"` and calls `ensure_bedrock_iam_secret()`.
4. `ensure_bedrock_iam_secret()` reads env vars based on a discriminator (`PROVIDER_KEY_PATH` value of `"iam"` or `"iam_role"`) and creates the `llmcreds` K8s secret with the appropriate keys.

### `ensure_bedrock_iam_secret()` Design

Modeled on `ensure_azure_entra_id_secret()` (lines 232–261 of `ols_installer.py`):

```
_BEDROCK_IAM_ENV_KEYS = ("BEDROCK_AWS_ACCESS_KEY_ID", "BEDROCK_AWS_SECRET_ACCESS_KEY")
_BEDROCK_ROLE_ENV_KEYS = ("BEDROCK_ROLE_AWS_ACCESS_KEY_ID", "BEDROCK_ROLE_AWS_SECRET_ACCESS_KEY", "BEDROCK_ROLE_ARN")
```

- If `creds == "iam"`: read `_BEDROCK_IAM_ENV_KEYS`, create secret with `aws_access_key_id` + `aws_secret_access_key`.
- If `creds == "iam_role"`: read `_BEDROCK_ROLE_ENV_KEYS`, create secret with `aws_access_key_id` + `aws_secret_access_key` + `role_arn`.
- Missing env vars: print warning and skip (same pattern as Azure Entra ID).

### `create_secrets()` Update

Add a branch at the top of `create_secrets()`:

```python
if provider_name.startswith("bedrock"):
    ensure_bedrock_iam_secret(creds)
    return
```

The `creds` parameter (sourced from `PROVIDER_KEY_PATH` in `run_suite`) acts as the discriminator (`"iam"` or `"iam_role"`).

## OLSConfig CR Templates

Three new files in `tests/config/operator_install/`:

### `olsconfig.crd.bedrock_anthropic.yaml`

```yaml
apiVersion: ols.openshift.io/v1alpha1
kind: OLSConfig
metadata:
  name: cluster
  labels:
    app.kubernetes.io/created-by: lightspeed-operator
    app.kubernetes.io/instance: olsconfig-sample
    app.kubernetes.io/managed-by: kustomize
    app.kubernetes.io/name: olsconfig
    app.kubernetes.io/part-of: lightspeed-operator
spec:
  llm:
    providers:
      - name: bedrock
        type: bedrock
        url: "https://bedrock-mantle.us-east-1.api.aws"
        credentialsSecretRef:
          name: llmcreds
        models:
          - name: anthropic.claude-opus-4-7
  ols:
    defaultModel: anthropic.claude-opus-4-7
    defaultProvider: bedrock
    deployment:
      replicas: 1
    disableAuth: false
    logLevel: DEBUG
    queryFilters:
      - name: foo_filter
        pattern: '\b(?:foo)\b'
        replaceWith: "deployment"
      - name: bar_filter
        pattern: '\b(?:bar)\b'
        replaceWith: "openshift"
    userDataCollection:
      feedbackDisabled: true
      transcriptsDisabled: true
```

### `olsconfig.crd.bedrock_openai.yaml`

Same as above with:
- `models[0].name: openai.gpt-5.4`
- `defaultModel: openai.gpt-5.4`

### `olsconfig.crd.bedrock_openai_tool_calling.yaml`

Same as `bedrock_openai.yaml` with `introspectionEnabled: true` added under `ols:` (this is the only difference that enables tool calling, matching the pattern in `olsconfig.crd.openai_tool_calling.yaml`).

## Test Suites

### `test-e2e-cluster-periodics.sh` Entries

| Suite ID | Test tags | Provider | PROVIDER_KEY_PATH | Model | OLS_CONFIG_SUFFIX |
|---|---|---|---|---|---|
| `bedrock_anthropic` | `not azure_entra_id and not certificates and not (tool_calling and not smoketest and not rag) and not byok1 and not byok2 and not quota_limits and not data_export` | `bedrock_anthropic` | `iam` | `anthropic.claude-opus-4-7` | `default` |
| `bedrock_openai` | same as above | `bedrock_openai` | `iam` | `openai.gpt-5.4` | `default` |
| `bedrock_anthropic_iam_role` | `smoketest` | `bedrock_anthropic` | `iam_role` | `anthropic.claude-opus-4-7` | `default` |
| `bedrock_openai_iam_role` | `smoketest` | `bedrock_openai` | `iam_role` | `openai.gpt-5.4` | `default` |
| `bedrock_openai_tool_calling` | `tool_calling` | `bedrock_openai` | `iam` | `openai.gpt-5.4` | `tool_calling` |

### `test-e2e-cluster.sh`

Mirror the periodics entries. Whether all five suites or a subset run on PR CI is a CI-config decision; the script should include all of them.

### CR Template Selection

The existing `adapt_ols_config.py` logic builds the CR filename as:
```
olsconfig.crd.{provider}  →  if suffix != "default": += _{suffix}
```

This does not work directly for Bedrock because the provider name is `bedrock` but the CR templates use `bedrock_anthropic` / `bedrock_openai`. The `run_suite` call controls the model via a CLI argument, but the CR template name is derived from `PROVIDER` alone.

**Solution:** The CR template filename must encode the model family. Pass `PROVIDER` as `bedrock_anthropic` or `bedrock_openai` in the `run_suite` call. Then in `create_secrets()` and `conftest.py`, strip the model suffix to get the actual provider name (`bedrock`) for credential handling. Alternatively, use `OLS_CONFIG_SUFFIX` to carry the model variant — but this conflicts with the existing suffix values (`default`, `tool_calling`, etc.).

The cleanest approach: pass `PROVIDER=bedrock_anthropic` or `PROVIDER=bedrock_openai`. In `create_secrets()`, match on `provider_name.startswith("bedrock")` rather than `provider_name == "bedrock"`. The CR template selection then works naturally: `olsconfig.crd.bedrock_anthropic.yaml`, `olsconfig.crd.bedrock_openai_tool_calling.yaml`, etc.

The `wait_for_ols` and test execution code only uses `PROVIDER` for display/logging — it does not parse the provider name for behavior.

## Delivery

### Two PRs

| PR | Repo | Content |
|---|---|---|
| 1 | `openshift/lightspeed-service` | CR templates, `ensure_bedrock_iam_secret()`, `create_secrets` bedrock branch, `run_suite` entries |
| 2 | `openshift/release` | Add five Bedrock env vars to existing periodic job Vault credential mapping |

### Merge Order

1. Operator PR #1749 (OLS-2605)
2. Service PR (this story)
3. Release repo PR

Service PR can merge before operator PR since the new `run_suite` entries will not be triggered until the release repo PR provides the Vault credentials. However, the CR templates reference `type: bedrock` which requires the operator CRD update.

## Files Changed

| File | Change |
|---|---|
| `tests/e2e/utils/ols_installer.py` | Add `ensure_bedrock_iam_secret()`, update `create_secrets()` |
| `tests/config/operator_install/olsconfig.crd.bedrock_anthropic.yaml` | New file |
| `tests/config/operator_install/olsconfig.crd.bedrock_openai.yaml` | New file |
| `tests/config/operator_install/olsconfig.crd.bedrock_openai_tool_calling.yaml` | New file |
| `tests/scripts/test-e2e-cluster-periodics.sh` | Add five `run_suite` calls |
| `tests/scripts/test-e2e-cluster.sh` | Mirror periodics entries |
