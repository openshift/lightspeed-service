# Hybrid inference example

Two KServe `InferenceService`s behind one `OLSConfig`, exposed as two OLS providers:

| OLS provider / model | Model | ServingRuntime | Image | Hardware |
|---|---|---|---|---|
| `cpu-small` | `ibm-granite/granite-4.2-3b-fp8` | `cpu-small-runtime` | `registry.redhat.io/rhaii/vllm-cpu-rhel9:3.5` | CPU |
| `gpu-large` | `ibm-granite/granite-4.2-8b-fp8` | `gpu-large-runtime` | `registry.redhat.io/rhaii/vllm-cuda-rhel9:3.5` | 1 GPU |

Each InferenceService sets `--served-model-name`, so `/v1/models` returns
`cpu-small` / `gpu-large` and OLS sends those names upstream.

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

oc run curl-test --rm -it --restart=Never -n openshift-lightspeed \
  --image=curlimages/curl -- \
  curl -s http://gpu-large-predictor.openshift-lightspeed.svc.cluster.local:8080/v1/models
```

Query through OLS:

```bash
curl -sk -H "Authorization: Bearer $(oc whoami -t)" \
  -H 'Content-Type: application/json' \
  "https://<ols-route>/v1/query" \
  -d '{"query":"what is a deployment?","provider":"gpu-large","model":"gpu-large"}'
```

## Routing by mode

OLS does not map `mode` to a provider/model; `mode` only selects the system prompt.
To send ASK traffic to `cpu-small` and TROUBLESHOOTING traffic to `gpu-large`, have
the client send `provider`/`model` per request, or put a proxy in front of OLS.
