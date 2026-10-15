# SkyNet Operations Guide

SkyNet is a fictional workload management product from Cyberdyne. This document
is synthetic content used exclusively to test OpenShift Lightspeed BYOK retrieval.

## Checking SkyNet status on OpenShift

Cyberdyne SkyNet runs on OpenShift as the skynet-controller deployment in the
cyberdyne-system namespace. To check the status of Cyberdyne SkyNet on OpenShift,
inspect that deployment:

```sh
oc get deployment skynet-controller -n cyberdyne-system
oc rollout status deployment/skynet-controller -n cyberdyne-system
```

SkyNet is ready when the deployment has its desired number of available replicas
and the rollout status command reports that it successfully rolled out.
