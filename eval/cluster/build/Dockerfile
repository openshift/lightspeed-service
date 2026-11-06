# Build context: this repo's eval/ tree and examples/hybrid-inference/cpu-small/inferenceservice.yaml.
# The eval sources are downloaded inside the build (EVAL_REPO_URL / EVAL_REF below), so no local
# checkout of lightspeed-evaluation is needed. Build it with the oc start-build command in README.md.
FROM registry.access.redhat.com/ubi9/python-312:latest

ARG EVAL_REPO_URL=https://github.com/RMalone8/lightspeed-evaluation
ARG EVAL_REF=fix/support-ols-streaming-responses-in-streaming-parser
# Set to a timestamp by stage_and_build.sh: the branch moves, so the value must change or a cached
# layer would serve an old clone.
ARG BUILD_ID=local

USER 0
RUN curl -fsSL https://mirror.openshift.com/pub/openshift-v4/clients/ocp/stable/openshift-client-linux.tar.gz \
    | tar -xz -C /usr/local/bin oc kubectl

RUN pip install --no-cache-dir --upgrade pip

RUN mkdir -p /opt/lightspeed-evaluation \
    && curl -fsSL "$EVAL_REPO_URL/archive/refs/heads/$EVAL_REF.tar.gz" \
       | tar -xz -C /opt/lightspeed-evaluation --strip-components=1 \
    && rm -rf /opt/lightspeed-evaluation/tests /opt/lightspeed-evaluation/.github \
    && printf 'repo: %s\nref: %s\ncommit: %s\nbuilt: %s\n' "$EVAL_REPO_URL" "$EVAL_REF" \
       "$(curl -fsSL "https://api.github.com/repos/${EVAL_REPO_URL#https://github.com/}/commits/$EVAL_REF" \
          | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])' 2>/dev/null || echo unknown)" \
       "$BUILD_ID" > /opt/lightspeed-evaluation/BUILD_REF

# lightspeed-evaluation pins datasets<=4.0.0, which pins fsspec to a version with a known High CVE.
# datasets 5.0.1 and fsspec 2026.6.0 are the fixed releases; pip prints a conflict message but installs.
RUN pip install --no-cache-dir /opt/lightspeed-evaluation pyyaml \
    && pip install --no-cache-dir "datasets>=5.0.1" "fsspec>=2026.6.0" "click>=8.3.3,<8.4" "huggingface-hub<2"

COPY eval /opt/hybrid/eval
COPY examples/hybrid-inference/cpu-small/inferenceservice.yaml \
     /opt/hybrid/examples/hybrid-inference/cpu-small/inferenceservice.yaml

RUN mkdir -p /results && chown -R 1001:0 /results /opt/hybrid && chmod -R g=u /results

USER 1001
WORKDIR /opt/hybrid
ENTRYPOINT ["python", "eval/scripts/run_experiments.py"]
