# vim: set filetype=dockerfile
ARG LIGHTSPEED_RAG_CONTENT_IMAGE=quay.io/redhat-user-workloads/crt-nshift-lightspeed-tenant/own-app-lightspeed-rag-content@sha256:6a26cfb029969de645486ccc7736e4f2266e284826618075510333a06c387b43
ARG BUILDER_BASE_IMAGE=registry.redhat.io/rhel9/python-312@sha256:46f883684d02cef2a7abb0c4124f18308ad920018d76c5c56f130dae02bfed05
ARG RUNTIME_BASE_IMAGE=registry.redhat.io/rhel9/python-312-minimal@sha256:804b928fd278fa03c2edf0352378eca73c8efcf665c6e0180e074340b9f22a50
FROM --platform=linux/amd64 ${LIGHTSPEED_RAG_CONTENT_IMAGE} AS lightspeed-rag-content

FROM --platform=$BUILDPLATFORM ${BUILDER_BASE_IMAGE} AS builder
ARG BUILDER_DNF_COMMAND=dnf
ARG APP_ROOT=/app-root

USER root

RUN ${BUILDER_DNF_COMMAND} install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs \
    gcc gcc-c++ cmake cargo

# UV_PYTHON_DOWNLOADS=0 : Disable Python interpreter downloads and use the system interpreter.
# UV_COMPILE_BYTECODE=0 : Disable bytecode compilation.
# UV_LINK_MODE=copy : Use copy mode for linking.
# MATURIN_NO_INSTALL_RUST=1 : Disable Rust installation.
ENV UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    MATURIN_NO_INSTALL_RUST=1

WORKDIR /app-root

# Add explicit files and directories
# (avoid accidental inclusion of local directories or env files or credentials)
COPY runner.py requirements.hashes.wheel.txt requirements.hashes.source.txt pyproject.toml uv.lock LICENSE README.md ./

COPY ols ./ols

# Install uv package manager
RUN pip install "uv>=0.8.15"

# Bundle additional dependencies for library mode.
# Source cachi2 environment for hermetic builds if available, otherwise use normal installation
# cachi2.env has these env vars:
# PIP_FIND_LINKS=/cachi2/output/deps/pip
# PIP_NO_INDEX=true
RUN if [ -f /cachi2/cachi2.env ]; then \
    . /cachi2/cachi2.env && \
    uv venv --seed --no-index --find-links ${PIP_FIND_LINKS} && \
    . .venv/bin/activate && \
    pip install --no-cache-dir --ignore-installed --no-index --find-links ${PIP_FIND_LINKS} --no-deps -r requirements.hashes.wheel.txt -r requirements.hashes.source.txt ;\
    else \
    uv sync --locked --no-dev --no-cache ;\
    fi

# Add executables from .venv to system PATH
ENV PATH="/app-root/.venv/bin:$PATH"

# Verify all dependencies are installed correctly
RUN echo "Verifying dependencies installation..." && \
    pip check && \
    python -c "import yaml, fastapi, langchain, llama_index, uvicorn, pydantic" && \
    echo "All dependencies installed and verified successfully!"

FROM ${RUNTIME_BASE_IMAGE}
ARG APP_ROOT=/app-root

WORKDIR /app-root

# PYTHONDONTWRITEBYTECODE 1 : disable the generation of .pyc
# PYTHONUNBUFFERED 1 : force the stdout and stderr streams to be unbuffered
# PYTHONCOERCECLOCALE 0, PYTHONUTF8 1 : skip legacy locales and use UTF-8 mode
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONCOERCECLOCALE=0 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=en_US.UTF-8 \
    LLAMA_INDEX_CACHE_DIR=/tmp/llama_index

COPY --from=builder /app-root/.venv .venv
COPY ols ./ols
COPY runner.py /app-root/runner.py
COPY --from=lightspeed-rag-content /rag/vector_db/ocp_product_docs ./vector_db/ocp_product_docs
COPY --from=lightspeed-rag-content /rag/embeddings_model ./embeddings_model

# this directory is checked by ecosystem-cert-preflight-checks task in Konflux
COPY LICENSE /licenses/

# Add executables from .venv to system PATH
ENV PATH="/app-root/.venv/bin:$PATH"

# Run the application
EXPOSE 8080
EXPOSE 8443
ENTRYPOINT ["python", "runner.py"]

LABEL io.k8s.display-name="OpenShift LightSpeed Service" \
    io.k8s.description="AI-powered OpenShift Assistant Service." \
    io.openshift.tags="openshift-lightspeed,ols" \
    description="Red Hat OpenShift Lightspeed Service" \
    summary="Red Hat OpenShift Lightspeed Service" \
    com.redhat.component=openshift-lightspeed-service \
    name="openshift-lightspeed/lightspeed-service-api-rhel9" \
    cpe="cpe:/a:redhat:openshift_lightspeed:1::el9" \
    vendor="Red Hat, Inc."


# no-root user is checked in Konflux
USER 1001
