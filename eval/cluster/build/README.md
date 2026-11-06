# Building the eval-runner image

The Job pulls the published image `quay.io/rmalone/lightspeed-eval-runner:latest`; nothing here is
needed to run experiments (see [`../README.md`](../README.md)).

| File | Purpose |
|---|---|
| `Dockerfile` | UBI9 Python 3.12, `oc`, `lightspeed-evaluation` and this repo's `eval/` tree |
| `build.yaml` | Binary BuildConfig that pushes to Quay |

The image carries the runner scripts, the system config, the datasets and `lightspeed-evaluation`
from [RMalone8/lightspeed-evaluation](https://github.com/RMalone8/lightspeed-evaluation) at `EVAL_REF`
(currently `fix/support-ols-streaming-responses-in-streaming-parser`), with the commit recorded in
`/opt/lightspeed-evaluation/BUILD_REF`.

## One-time setup

1. In Quay, create a robot account with **write** access to `rmalone/lightspeed-eval-runner`.
2. Give the cluster its credentials (the robot account, not your own login) and apply the BuildConfig:
   ```bash
   oc create secret docker-registry quay-push -n openshift-lightspeed \
     --docker-server=quay.io --docker-username='rmalone+<robot-name>' --docker-password='<robot-token>'
   oc apply -f eval/cluster/build/build.yaml
   ```
3. Set the repository to **public** in the Quay UI after the first push.

## Each release

From the repository root:
```bash
oc start-build eval-runner -n openshift-lightspeed --from-dir=. \
  --exclude='(^|/)(\.git|\.venv|embeddings_model|node_modules|__pycache__|\.env[^/]*)(/|$)' \
  --build-arg=BUILD_ID="$(date -u +%Y%m%dT%H%M%SZ)" \
  --follow
```

Rebuild when the runner scripts, system config or datasets change, or when the branch moves.
`--exclude` only keeps the upload small; `BUILD_ID` records the build time and stops a cached layer
from serving an older copy of the moving branch. For another branch or tag, add
`--build-arg=EVAL_REF=<branch-or-tag>`.

The Dockerfile installs `datasets` 5.x and `fsspec` 2026.x over the repo's pins, because the pinned
`fsspec` has a known High CVE.
