# vim: set filetype=dockerfile
ARG LIGHTSPEED_RAG_CONTENT_DIGEST=sha256:d15bf56776c40a8709b0e648e3b0f043de63b24ad8f59eeea6f8d965dfcbe4e3

FROM quay.io/openshift/lightspeed-rag-content@${LIGHTSPEED_RAG_CONTENT_DIGEST} as lightspeed-rag-content

FROM registry.access.redhat.com/ubi9/ubi-minimal

ARG VERSION
ARG APP_ROOT=/app-root

RUN microdnf install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs \
    python3.11 python3.11-devel python3.11-pip jq shadow-utils \
    && microdnf clean all --enablerepo='*'

# PYTHONDONTWRITEBYTECODE 1 : disable the generation of .pyc
# PYTHONUNBUFFERED 1 : force the stadout and stderr streams to be unbufferred
# PYTHONCOERCECLOCALE 0, PYTHONUTF8 1 : skip lgeacy locales and use UTF-8 mode
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONCOERCECLOCALE=0 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=en_US.UTF-8 \
    PIP_NO_CACHE_DIR=off

WORKDIR ${APP_ROOT}
# Add explicit files and directories
# (avoid accidental inclusion of local directories or env files or credentials)
COPY ols ./ols
COPY pyproject.toml pdm.lock runner.py ./
COPY --from=lightspeed-rag-content /rag/vector_db/ocp_product_docs ./vector_db/ocp_product_docs
COPY --from=lightspeed-rag-content /rag/embeddings_model ./embeddings_model

RUN pip3.11 install --no-cache-dir --upgrade pip pdm \
    && pdm config python.use_venv false \
    && pdm sync --global --prod -p ${APP_ROOT}

# Run the application
EXPOSE 8080
EXPOSE 8443
CMD ["python3.11", "runner.py"]

LABEL io.k8s.display-name="OpenShift LightSpeed Service" \
      io.k8s.description="AI-powered OpenShift Assistant Service." \
      io.openshift.tags="openshift-lightspeed,ols"
