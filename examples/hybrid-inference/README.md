# Hybrid inference example

Two KServe `InferenceService`s behind a [Praxis](https://github.com/praxis-proxy/ai)
gateway, exposed to OLS as a single provider. OLS tags each LLM request with an
`X-OLS-Mode` header and Praxis routes on it:

| OLS `mode` | `X-OLS-Mode` | Backend | Model | Hardware |
|---|---|---|---|---|
| `ask` | `ask` (or absent) | `cpu-small` | `ibm-granite/granite-4.2-3b-fp8` | CPU |
| `troubleshooting` | `troubleshooting` | `gpu-large` | `ibm-granite/granite-4.2-8b-fp8` | 1 GPU |

Both InferenceServices set `--served-model-name=granite` and `--max-model-len=32768`,
so OLS sees one model (`granite`, 32k context) and the routing is invisible to it.
Requests without the header, such as the OLS readiness check, fall through to `cpu-small`.

Praxis (`praxis/config.yaml`) also answers `GET /healthz` and `GET /v1/models` itself
with static responses, so neither depends on a backend being up. OLS uses a single
placeholder credential (`llmcreds-praxis`) for the gateway.

## Prerequisites

- Lightspeed operator in `openshift-lightspeed`.
- RHOAI/KServe installed.
- A node labeled `inference=cpu-small` for `cpu-small`, and a node labeled
  `inference=gpu-large` with one GPU for `gpu-large`. The manifests also tolerate an
  `inference=<value>:NoSchedule` taint on those nodes.

## Apply and verify

```bash
oc apply -k examples/hybrid-inference

oc get servingruntime,inferenceservice -n openshift-lightspeed
oc get pods -n openshift-lightspeed -l serving.kserve.io/inferenceservice

oc get deploy,svc -n openshift-lightspeed praxis

oc run curl-test --rm -it --restart=Never -n openshift-lightspeed \
  --image=curlimages/curl -- \
  curl -s http://praxis.openshift-lightspeed.svc.cluster.local:8080/v1/models
```

Query through OLS:

```bash
curl -sk -H "Authorization: Bearer $(oc whoami -t)" \
  -H 'Content-Type: application/json' \
  "https://<ols-route>/v1/query" \
  -d '{"query":"what is a deployment?","mode":"ask"}'

curl -sk -H "Authorization: Bearer $(oc whoami -t)" \
  -H 'Content-Type: application/json' \
  "https://<ols-route>/v1/query" \
  -d '{"query":"why is my pod CrashLooping?","mode":"troubleshooting"}'
```

Confirm routing from the backend access logs:

```bash
oc logs -n openshift-lightspeed -l serving.kserve.io/inferenceservice=cpu-small --tail=20
oc logs -n openshift-lightspeed -l serving.kserve.io/inferenceservice=gpu-large --tail=20
```

## Requirements on the OLS image

The `X-OLS-Mode` header is sent by the OLS service itself (`DocsSummarizer`), for
`rhoai_vllm` providers only. The OLS image deployed by the operator must include
that change.

## Building the OLS image on the cluster

`build/buildconfig.yaml` builds this repo's `Containerfile` on an amd64 node and
pushes the result to the internal registry. It is not part of the kustomization.
The build clones `ref` from GitHub, so push your changes there first.

```bash
oc apply -f examples/hybrid-inference/build/buildconfig.yaml
oc start-build lightspeed-service -n openshift-lightspeed --follow
```

Point the operator at the image by replacing the `--service-image=` arg in the
operator's ClusterServiceVersion:

```bash
oc get csv -n openshift-lightspeed
oc get csv <csv> -n openshift-lightspeed -o json \
  | jq '.spec.install.spec.deployments[0].spec.template.spec.containers[0].args'

oc patch csv <csv> -n openshift-lightspeed --type=json -p '[
  {"op":"replace",
   "path":"/spec/install/spec/deployments/0/spec/template/spec/containers/0/args/<N>",
   "value":"--service-image=image-registry.openshift-image-registry.svc:5000/openshift-lightspeed/lightspeed-service:hybrid"}]'
```

An operator upgrade replaces the CSV and drops the patch. After rebuilding under the same
tag, run `oc rollout restart deploy/lightspeed-app-server -n openshift-lightspeed`.
