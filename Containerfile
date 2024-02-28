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

RUN pip3.11 install --no-cache-dir --upgrade pip pdm \
    && pdm config python.use_venv false \
    && pdm install --global --prod --frozen-lockfile -p ${APP_ROOT}

# Run the application
EXPOSE 8080
CMD ["python3.11", "runner.py"]

LABEL io.k8s.display-name="OpenShift LightSpeed Service" \
      io.k8s.description="AI-powered OpenShift Assistant Service." \
      io.openshift.tags="openshift-lightspeed,ols"
