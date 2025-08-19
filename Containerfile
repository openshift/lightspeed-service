# vim: set filetype=dockerfile
ARG LIGHTSPEED_RAG_CONTENT_IMAGE=quay.io/redhat-user-workloads/crt-nshift-lightspeed-tenant/own-app-lightspeed-rag-content@sha256:5a684b6b75ba0e5c34b322494d1ca41f8c3e5d1b753d174129d0d8a7536d2e38

# Get RAG content from the existing RAG image
FROM --platform=linux/amd64 ${LIGHTSPEED_RAG_CONTENT_IMAGE} as lightspeed-rag-content

# Use lightspeed-stack as the base image instead of building our own service
FROM --platform=linux/amd64 quay.io/lightspeed-core/lightspeed-stack:dev-latest

# Copy RAG content from the RAG image into the lightspeed-stack image
COPY --from=lightspeed-rag-content /rag/vector_db/ocp_product_docs /app-root/vector_db/ocp_product_docs
COPY --from=lightspeed-rag-content /rag/embeddings_model /app-root/embeddings_model

# this directory is checked by ecosystem-cert-preflight-checks task in Konflux
COPY LICENSE /licenses/

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
