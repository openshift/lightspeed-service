# vim: set filetype=dockerfile
ARG BUILDER_BASE_IMAGE=registry.redhat.io/rhel9/python-312@sha256:46f883684d02cef2a7abb0c4124f18308ad920018d76c5c56f130dae02bfed05
ARG RUNTIME_BASE_IMAGE=registry.redhat.io/rhel9/python-312-minimal@sha256:804b928fd278fa03c2edf0352378eca73c8efcf665c6e0180e074340b9f22a50

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

# Step 1: Copy only dependency metadata (not source code).
# LICENSE and README.md are needed by hatchling for metadata resolution.
COPY pyproject.toml uv.lock .konflux/requirements.hashes.wheel.txt .konflux/requirements.hashes.source.txt .konflux/requirements.hashes.wheel.pypi.txt .konflux/requirements.hermetic.txt .konflux/rank_bm25_version.py LICENSE README.md ./

# Step 2: Install dependencies only (cached unless pyproject.toml/uv.lock change).
# Source cachi2 environment for hermetic builds if available, otherwise use normal installation.
# cachi2.env has these env vars:
# PIP_FIND_LINKS=/cachi2/output/deps/pip
# PIP_NO_INDEX=true
RUN if [ -f /cachi2/cachi2.env ]; then \
    . /cachi2/cachi2.env && \
    pip install --no-cache-dir --no-index --find-links ${PIP_FIND_LINKS} uv && \
    uv venv && \
    for f in requirements.hashes.wheel.txt requirements.hashes.source.txt requirements.hashes.wheel.pypi.txt; do \
        sed -i '/^--index-url /d' "$f"; \
    done && \
    for sdist in ${PIP_FIND_LINKS}/rank_bm25-*.tar.gz; do \
        if [ -f "$sdist" ]; then \
            tmpdir=$(mktemp -d) && \
            tar -xzf "$sdist" -C "$tmpdir" && \
            setupdir=$(find "$tmpdir" -name setup.py -exec dirname {} \;) && \
            cp rank_bm25_version.py "$setupdir/version.py" && \
            (cd "$tmpdir" && tar -czf "$sdist" *) && \
            rm -rf "$tmpdir"; \
        fi; \
    done && \
    sed -i '/^rank-bm25/,/^[^ ]/{ /^rank-bm25/d; /^    --hash/d; }' requirements.hashes.source.txt && \
    uv pip install --python .venv/bin/python --no-cache --no-index --find-links ${PIP_FIND_LINKS} --no-deps -r requirements.hashes.wheel.txt -r requirements.hashes.source.txt -r requirements.hashes.wheel.pypi.txt rank-bm25 ;\
    else \
    pip install "uv>=0.8.15" && \
    uv sync --locked --no-dev --no-cache --no-install-project ;\
    fi

# Step 3: Copy source code (only this layer rebuilds on code changes).
COPY runner.py ./
COPY ols ./ols

# Step 4: Install the project package itself (non-hermetic only).
RUN if [ ! -f /cachi2/cachi2.env ]; then \
    uv sync --locked --no-dev --no-cache ;\
    fi

# Add executables from .venv to system PATH
ENV PATH="/app-root/.venv/bin:$PATH"

# Pre-warm tiktoken encoding cache to a stable, version-agnostic location.
RUN src=$(find .venv/lib -path "*tiktoken_cache" -type d | head -1) && \
    mkdir -p /app-root/.tiktoken_cache && \
    if [ -n "$src" ]; then cp "$src"/[0-9a-f]* /app-root/.tiktoken_cache/; fi

# Verify all dependencies are installed correctly
RUN echo "Verifying dependencies installation..." && \
    pip check && \
    python -c "import yaml, fastapi, langchain, llama_index, uvicorn, pydantic" && \
    TIKTOKEN_CACHE_DIR=/app-root/.tiktoken_cache python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" && \
    echo "All dependencies installed and verified successfully!"

# Step 5: Reassemble embedding model safetensors from compressed chunks and validate.
COPY --chmod=775 embeddings_model ./embeddings_model
RUN for model_dir in all-mpnet-base-v2 granite-embedding-30m-english; do \
      cat "embeddings_model/${model_dir}"/model.safetensors.tar.gz.* | \
        tar xzf - --no-same-owner -C "embeddings_model/${model_dir}" || \
      { echo "ERROR: failed to extract embeddings_model/${model_dir}/model.safetensors from chunks" ; exit 1 ; } && \
      rm -f "embeddings_model/${model_dir}"/model.safetensors.tar.gz.* && \
      python3 -c "import safetensors; safetensors.safe_open('embeddings_model/${model_dir}/model.safetensors', framework='pt'); print('OK:', '${model_dir}')" || \
      { echo "ERROR: corrupt safetensors file: embeddings_model/${model_dir}/model.safetensors" ; exit 1 ; } ; \
    done

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
    LLAMA_INDEX_CACHE_DIR=/app-root/.cache/llama_index \
    TIKTOKEN_CACHE_DIR=/app-root/.tiktoken_cache \
    HF_HOME=/app-root/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY --from=builder /app-root/.venv .venv
COPY --from=builder /app-root/.tiktoken_cache .tiktoken_cache
COPY ols ./ols
COPY runner.py /app-root/runner.py
COPY --chmod=775 --from=builder /app-root/embeddings_model ./embeddings_model

# Pre-populate HuggingFace cache so models can be loaded by ID with TRANSFORMERS_OFFLINE=1.
# Two cache trees are needed: the standard HF hub cache (used by huggingface_hub and
# transformers when no cache_dir override is given) and the llama_index cache (passed as
# cache_folder to SentenceTransformer by llama_index's HuggingFaceEmbedding wrapper).
USER root
RUN for model_dir in all-mpnet-base-v2 granite-embedding-30m-english; do \
    case "$model_dir" in \
      all-mpnet-base-v2) hf_id="sentence-transformers--all-mpnet-base-v2" ;; \
      granite-embedding-30m-english) hf_id="ibm-granite--granite-embedding-30m-english" ;; \
    esac && \
    for cache_root in /app-root/.cache/huggingface/hub /app-root/.cache/llama_index; do \
      repo_dir="${cache_root}/models--${hf_id}" && \
      mkdir -p "$repo_dir/snapshots/local" "$repo_dir/refs" && \
      printf '%s' "local" > "$repo_dir/refs/main" && \
      ln -sf "/app-root/embeddings_model/${model_dir}"/* "$repo_dir/snapshots/local/" ; \
    done ; \
    done && \
    chown -R 1001:0 /app-root/.cache

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
