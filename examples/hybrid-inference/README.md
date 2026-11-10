# Hybrid inference example

Two KServe `InferenceService`s behind a [Praxis](https://github.com/praxis-proxy/ai) gateway,
exposed to OLS as one provider. OLS tags each request with `X-OLS-Mode` and Praxis routes on it:

| OLS `mode` | `X-OLS-Mode` | Backend | Model | Hardware |
|---|---|---|---|---|
| `ask` | `ask` | `cpu-small` | `Qwen/Qwen-AgentWorld-35B-A3B` | CPU |
| `troubleshooting` | `troubleshooting` or absent | `gpu-large` | `ibm-granite/granite-4.2-8b-fp8` | 1 GPU |

Both serve `--served-model-name=hybrid-model --max-model-len=32768`, so OLS sees one model and the routing
is invisible to it. Only an explicit `X-OLS-Mode: ask` reaches `cpu-small`; requests without the
header, such as the readiness check, go to `gpu-large`, whose 30s timeout the CPU model cannot meet.
Praxis answers `GET /healthz` and `GET /v1/models` itself, and OLS uses one placeholder credential
(`llmcreds-praxis`). The `praxis-config` ConfigMap carries a content-hash suffix, so `oc apply -k`
after editing `praxis/config.yaml` rolls the Deployment.

## Prerequisites

Lightspeed operator and RHOAI/KServe installed, a node labeled `inference=cpu-small`, and one labeled
`inference=gpu-large` with a GPU. The manifests tolerate the matching `NoSchedule` taints:

```bash
oc label node <cpu-node> inference=cpu-small
oc adm taint nodes <cpu-node> inference=cpu-small:NoSchedule
oc label node <gpu-node> inference=gpu-large
oc adm taint nodes <gpu-node> inference=gpu-large:NoSchedule
```

## Apply and verify

```bash
oc apply -k examples/hybrid-inference
oc get servingruntime,inferenceservice -n openshift-lightspeed
oc get deploy,svc -n openshift-lightspeed praxis
oc run curl-test --rm -it --restart=Never -n openshift-lightspeed \
  --image=curlimages/curl -- \
  curl -s http://praxis.openshift-lightspeed.svc.cluster.local:8080/v1/models
```

## Gated models

Models such as Llama need a Hugging Face token: accept the licence on the model page, create a read
token, and store it as `hf-token`/`HF_TOKEN`. Both manifests read it as `optional: true`, so ungated
models need nothing, and a `401`/`403` during download means the token is missing or not approved.

```bash
oc create secret generic hf-token -n openshift-lightspeed --from-literal=HF_TOKEN="$HF_TOKEN"
```

Experiments mark such a model `gated: true` in `eval/cluster/experiments.yaml`
([eval/cluster/README.md](../../eval/cluster/README.md)).

## Custom OLS image

`X-OLS-Mode` is sent by the OLS service itself (`DocsSummarizer`) for `rhoai_vllm` providers, so the
deployed image must contain that change; the operator passes `--service-image=` to
`lightspeed-app-server`.

A GitHub Actions workflow (`.github/workflows/build-ols-image.yaml`) builds the image for every PR
to `main` without pushing it, and on merge pushes `ghcr.io/rmalone8/lightspeed-service:latest` plus
a tag for the commit SHA.

To build on the cluster instead (`build/buildconfig.yaml` clones `ref` from GitHub, so push first):

```bash
oc apply -f examples/hybrid-inference/build/buildconfig.yaml
oc start-build lightspeed-service -n openshift-lightspeed --follow
```

With OLM, patch the CSV instead of the Deployment, selecting the
`lightspeed-operator-controller-manager` deployment:

```bash
NS=openshift-lightspeed
IMG=ghcr.io/rmalone8/lightspeed-service:latest   # or, for the on-cluster build: image-registry.openshift-image-registry.svc:5000/$NS/lightspeed-service:hybrid
CSV=$(oc get csv -n $NS -o name | grep lightspeed-operator | head -1 | cut -d/ -f2)
DEP=$(oc get csv $CSV -n $NS -o json | jq \
  '.spec.install.spec.deployments | map(.name == "lightspeed-operator-controller-manager") | index(true)')
IDX=$(oc get csv $CSV -n $NS -o json | jq \
  ".spec.install.spec.deployments[$DEP].spec.template.spec.containers[0].args
   | map(startswith(\"--service-image=\")) | index(true)")
oc patch csv $CSV -n $NS --type=json -p "[
  {\"op\":\"replace\",
   \"path\":\"/spec/install/spec/deployments/$DEP/spec/template/spec/containers/0/args/$IDX\",
   \"value\":\"--service-image=$IMG\"}]"
```

The operator then updates `lightspeed-app-server`; if it does not, force a reconcile with
`oc annotate olsconfig cluster -n $NS touch=$(date +%s) --overwrite`. An operator upgrade replaces the
CSV, so re-apply the patch. A new image under the same tag is not picked up by itself: the pull
policy is `Always`, but only when the pod restarts, so restart the app server after each build:

```bash
oc delete pod -n $NS -l app.kubernetes.io/name=lightspeed-service-api
```

Confirm the pod runs the new digest:

```bash
oc get pods -n $NS -o json | jq -r '.items[]
  | select(.metadata.name | startswith("lightspeed-app-server"))
  | .metadata.name as $p | .status.containerStatuses[]
  | "\($p) \(.name) \(.imageID)"'
```

## Test the routing

Send one request per mode to Praxis, then through OLS:

```bash
for MODE in ask troubleshooting; do
  oc run curl-$MODE --rm -it --restart=Never -n openshift-lightspeed \
    --image=curlimages/curl -- \
    curl -s -H "X-OLS-Mode: $MODE" -H 'Content-Type: application/json' \
    http://praxis.openshift-lightspeed.svc.cluster.local:8080/v1/chat/completions \
    -d '{"model":"hybrid-model","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
done

ROUTE=$(oc get route -n openshift-lightspeed -o jsonpath='{.items[0].spec.host}')
for MODE in ask troubleshooting; do
  curl -sk -H "Authorization: Bearer $(oc whoami -t)" -H 'Content-Type: application/json' \
    "https://$ROUTE/v1/query" -d "{\"query\":\"what is a deployment?\",\"mode\":\"$MODE\"}"
done
```

Follow both vLLM logs and confirm `ask` appears only in `cpu-small` and `troubleshooting` only in
`gpu-large` (a troubleshooting query makes several calls):

```bash
oc logs -f -n openshift-lightspeed -l serving.kserve.io/inferenceservice=cpu-small -c kserve-container
oc logs -f -n openshift-lightspeed -l serving.kserve.io/inferenceservice=gpu-large -c kserve-container
```

If everything lands on `cpu-small`, the header is not reaching Praxis: check that the app server runs
your image and that the provider type is `rhoai_vllm`.
