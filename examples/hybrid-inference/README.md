# Hybrid inference example

Two KServe `InferenceService`s behind a [Praxis](https://github.com/praxis-proxy/ai)
gateway, exposed to OLS as a single provider. OLS tags each LLM request with an
`X-OLS-Mode` header and Praxis routes on it:

| OLS `mode` | `X-OLS-Mode` | Backend | Model | Hardware |
|---|---|---|---|---|
| `ask` | `ask` | `cpu-small` | `ibm-granite/granite-4.2-3b-fp8` | CPU |
| `troubleshooting` | `troubleshooting` (or absent) | `gpu-large` | `ibm-granite/granite-4.2-8b-fp8` | 1 GPU |

Both InferenceServices set `--served-model-name=granite` and `--max-model-len=32768`,
so OLS sees one model (`granite`, 32k context) and the routing is invisible to it.
Requests without the header, such as the OLS readiness check, fall through to `gpu-large`;
only an explicit `X-OLS-Mode: ask` goes to `cpu-small`. The readiness check sends an
unconstrained request (up to 4096 tokens) that is likely too slow on the CPU model for its
30s timeout.

Praxis (`praxis/config.yaml`) also answers `GET /healthz` and `GET /v1/models` itself
with static responses, so neither depends on a backend being up. OLS uses a single
placeholder credential (`llmcreds-praxis`) for the gateway.

The `praxis-config` ConfigMap gets a content-hash suffix from kustomize, so editing
`praxis/config.yaml` and running `oc apply -k` renames the ConfigMap and rolls the Praxis
Deployment automatically. The Secret keeps a fixed name because `olsconfig.yaml` refers to
it by name. Old hashed ConfigMaps are not pruned; delete them if they pile up.

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

## Using a custom OLS image

The `X-OLS-Mode` header is sent by the OLS service itself (`DocsSummarizer`), for
`rhoai_vllm` providers only, so the OLS image the operator deploys must include that
change. The operator takes the image as a `--service-image=` argument on its
controller-manager and applies it to the `lightspeed-app-server` Deployment when it
reconciles.

### 1. Build the image on the cluster

`build/buildconfig.yaml` builds this repo's `Containerfile` on an amd64 node and
pushes the result to the internal registry as `lightspeed-service:hybrid`. It is not
part of the kustomization. The build clones `ref` (the branch) from GitHub, so push
your changes there first.

```bash
oc apply -f examples/hybrid-inference/build/buildconfig.yaml
oc start-build lightspeed-service -n openshift-lightspeed --follow
```

### 2. Point the operator at the image

When the operator is installed through OLM, OLM owns the operator Deployment and reverts
direct edits to it. The ClusterServiceVersion (CSV) is the source of truth, so patch the
`--service-image=` arg there. The CSV can hold more than one deployment (the agentic
operator is another), so select the one named `lightspeed-operator-controller-manager`
instead of assuming an index:

```bash
NS=openshift-lightspeed
IMG=image-registry.openshift-image-registry.svc:5000/$NS/lightspeed-service:hybrid
CSV=$(oc get csv -n $NS -o name | grep lightspeed-operator | head -1 | cut -d/ -f2)

DEP=$(oc get csv $CSV -n $NS -o json | jq \
  '.spec.install.spec.deployments | map(.name == "lightspeed-operator-controller-manager") | index(true)')
IDX=$(oc get csv $CSV -n $NS -o json | jq \
  ".spec.install.spec.deployments[$DEP].spec.template.spec.containers[0].args
   | map(startswith(\"--service-image=\")) | index(true)")
echo "csv=$CSV deployment=$DEP arg=$IDX"    # both numbers must be set, not null

oc patch csv $CSV -n $NS --type=json -p "[
  {\"op\":\"replace\",
   \"path\":\"/spec/install/spec/deployments/$DEP/spec/template/spec/containers/0/args/$IDX\",
   \"value\":\"--service-image=$IMG\"}]"
```

OLM restarts the operator pod. The operator then updates `lightspeed-app-server`. If the
app server does not change after the new operator pod is ready, trigger a reconcile:

```bash
oc annotate olsconfig cluster -n $NS touch=$(date +%s) --overwrite
```

### 3. Verify the running image

```bash
# Deployment spec: one container should show your image
oc get deploy lightspeed-app-server -n $NS \
  -o jsonpath='{range .spec.template.spec.containers[*]}{.name}{"  "}{.image}{"\n"}{end}'

# The digest of the build...
oc get istag lightspeed-service:hybrid -n $NS -o jsonpath='{.image.metadata.name}{"\n"}'

# ...should match the digest the pod is running
oc get pods -n $NS -o json | jq -r '.items[]
  | select(.metadata.name | startswith("lightspeed-app-server"))
  | .metadata.name as $p | .status.containerStatuses[]
  | "\($p) \(.name) \(.imageID)"'
```

An operator upgrade replaces the CSV and drops the patch. After rebuilding under the same
tag, run `oc rollout restart deploy/lightspeed-app-server -n $NS`.

## Test the routing

Check that the Praxis pod is running and the predictors were rolled to the new args
(`granite`, 32k) before testing; `oc apply -k` changes them.

```bash
oc get pods -n openshift-lightspeed | grep -E 'praxis|predictor'
```

### 1. Praxis alone

This isolates the gateway from OLS. Send one request per mode directly to Praxis:

```bash
for MODE in ask troubleshooting; do
  oc run curl-$MODE --rm -it --restart=Never -n openshift-lightspeed \
    --image=curlimages/curl -- \
    curl -s -H "X-OLS-Mode: $MODE" -H 'Content-Type: application/json' \
    http://praxis.openshift-lightspeed.svc.cluster.local:8080/v1/chat/completions \
    -d '{"model":"granite","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
done
```

### 2. Through OLS

```bash
ROUTE=$(oc get route -n openshift-lightspeed -o jsonpath='{.items[0].spec.host}')
TOKEN=$(oc whoami -t)

for MODE in ask troubleshooting; do
  curl -sk -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
    "https://$ROUTE/v1/query" \
    -d "{\"query\":\"what is a deployment?\",\"mode\":\"$MODE\"}"
done
```

`oc get route` may match more than one route. If the host is wrong, pick the OLS one by
name.

### 3. See which backend served each request

vLLM logs each request on the serving container (`kserve-container`). Run these, send
a request, and see which pod logs a `POST /v1/chat/completions`:

```bash
oc logs -f -n openshift-lightspeed -l serving.kserve.io/inferenceservice=cpu-small -c kserve-container
oc logs -f -n openshift-lightspeed -l serving.kserve.io/inferenceservice=gpu-large -c kserve-container
```

Expected: `ask` shows up only in the `cpu-small` log and `troubleshooting` only in the
`gpu-large` log. A troubleshooting query makes several calls (the answer plus any tool
and classifier calls), so expect more than one line on `gpu-large`.

If every request lands on `cpu-small`, the header is not reaching Praxis: check that the
app server is on your image (step 3 above) and that the provider type is `rhoai_vllm`.
