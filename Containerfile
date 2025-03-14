# vim: set filetype=dockerfile
ARG LIGHTSPEED_RAG_CONTENT_IMAGE=quay.io/openshift-lightspeed/lightspeed-rag-content@sha256:1646c92afe6a235fadc36e37480abf0c96aeb94eb08812da008ac8ecc56658e8
ARG HERMETIC=false

FROM ${LIGHTSPEED_RAG_CONTENT_IMAGE} as lightspeed-rag-content

FROM registry.redhat.io/ubi9/ubi-minimal@sha256:8b314e254e9ab9a7a08b675fcfd3ed66a2943eeda7b26395210d451569976b9b
ARG HERMETIC=false
ARG VERSION
ARG APP_ROOT=/app-root

RUN microdnf install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs \
    python3.11 python3.11-devel python3.11-pip

# conditional installation of OpenShift CLI
ENV HERMETIC=$HERMETIC
RUN if [ "$HERMETIC" == "true" ]; then \
      microdnf install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs openshift-clients; \
    else \
      OC_CLIENT_TAR_GZ=openshift-client-linux-amd64-rhel9-4.17.9.tar.gz; \
      microdnf install -y tar gzip && \
      curl -LO "https://mirror.openshift.com/pub/openshift-v4/x86_64/clients/ocp/4.17.9/${OC_CLIENT_TAR_GZ}" && \
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
    PIP_NO_CACHE_DIR=off

WORKDIR /app-root

COPY --from=lightspeed-rag-content /rag/vector_db/ocp_product_docs ./vector_db/ocp_product_docs
COPY --from=lightspeed-rag-content /rag/embeddings_model ./embeddings_model

# Add explicit files and directories
# (avoid accidental inclusion of local directories or env files or credentials)
COPY runner.py requirements.txt ./

RUN pip3.11 install --upgrade pip
RUN for a in 1 2 3 4 5; do pip3.11 install --no-cache-dir -r requirements.txt && break || sleep 15; done

COPY ols ./ols

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
      name=openshift-lightspeed-service \
      vendor="Red Hat, Inc."


# no-root user is checked in Konflux
USER 1001
