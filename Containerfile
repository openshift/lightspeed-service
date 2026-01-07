# vim: set filetype=dockerfile
ARG LIGHTSPEED_RAG_CONTENT_IMAGE=quay.io/redhat-user-workloads/crt-nshift-lightspeed-tenant/own-app-lightspeed-rag-content@sha256:a5aa44be470ce44429496ed02b690f0391936ee94dc9e3eef9a9f5a8fd48bc89
ARG HERMETIC=false

FROM --platform=linux/amd64 ${LIGHTSPEED_RAG_CONTENT_IMAGE} as lightspeed-rag-content

FROM --platform=$BUILDPLATFORM registry.redhat.io/ubi9/ubi-minimal:latest
ARG HERMETIC=false
ARG VERSION
ARG APP_ROOT=/app-root
ARG BUILDARCH

RUN microdnf install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs \
    python3.11 python3.11-devel python3.11-pip

# conditional installation of OpenShift CLI

ENV BUILDARCH=${BUILDARCH}
ENV HERMETIC=${HERMETIC}
RUN if [ "$HERMETIC" == "true" ]; then \
      microdnf install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs openshift-clients; \
    else \
      OC_CLIENT_TAR_GZ=openshift-client-linux-${BUILDARCH}-rhel9-4.17.16.tar.gz; \
      microdnf install -y tar gzip && \
      curl -LO "https://mirror.openshift.com/pub/openshift-v4/x86_64/clients/ocp/4.17.16/${OC_CLIENT_TAR_GZ}" && \
      tar xvfz ${OC_CLIENT_TAR_GZ} -C /usr/local/bin && \
      rm -f ${OC_CLIENT_TAR_GZ} && \
      chmod +x /usr/local/bin/oc && \
      microdnf remove -y tar gzip; \
    fi

RUN oc version --client

# PYTHONDONTWRITEBYTECODE 1 : disable the generation of .pyc
# PYTHONUNBUFFERED 1 : force the stdout and stderr streams to be unbuffered
# PYTHONCOERCECLOCALE 0, PYTHONUTF8 1 : skip legacy locales and use UTF-8 mode
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONCOERCECLOCALE=0 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=en_US.UTF-8 \
    PIP_NO_CACHE_DIR=off \
    LLAMA_INDEX_CACHE_DIR=/tmp/llama_index

WORKDIR /app-root

COPY --from=lightspeed-rag-content /rag/vector_db/ocp_product_docs ./vector_db/ocp_product_docs
COPY --from=lightspeed-rag-content /rag/embeddings_model ./embeddings_model

# Add explicit files and directories
# (avoid accidental inclusion of local directories or env files or credentials)
COPY runner.py requirements.txt ./

RUN pip3.11 install --upgrade pip
RUN pip3.11 install --no-cache-dir --ignore-installed -r requirements.txt

# Verify all dependencies are installed correctly
RUN echo "Verifying dependencies installation..." && \
    pip3.11 check && \
    python3.11 -c "import yaml, fastapi, langchain, llama_index, uvicorn, pydantic" && \
    echo "All dependencies installed and verified successfully!"

COPY ols ./ols
COPY mcp_local ./mcp_local

# this directory is checked by ecosystem-cert-preflight-checks task in Konflux
COPY LICENSE /licenses/

# Run the application
EXPOSE 8080
EXPOSE 8443
CMD ["python3.11", "runner.py"]

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
