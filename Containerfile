FROM registry.access.redhat.com/ubi9/python-311:latest

ARG VERSION
ARG APP_ROOT=/app-root

RUN microdnf install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs \
    jq shadow-utils \
    && microdnf clean all --enablerepo='*' \
    && useradd -r -u 1001 -g 0 -m -c "Default Application User" -d ${APP_ROOT} -s /bin/bash default

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
COPY modules ./modules
COPY tools ./tools
COPY logs ./logs
COPY src ./src
COPY ./ols.py ./requirements.txt ./
RUN chown -R 1001:0 ${APP_ROOT}

# default user for Python app
USER 1001
RUN python3.11 -m venv ${APP_ROOT}/venv \
    && source ${APP_ROOT}/venv/bin/activate \
    # Install wheel for optimized dependency installation \
    # avoid using or leaving cache on final image \
    && pip install --no-cache-dir --upgrade pip wheel \
    && pip install --no-cache-dir --upgrade -r requirements.txt \
    # The following echo adds the unset command for the variables set below to the \
    # venv activation script. This is inspired from scl_enable script and prevents \
    # the virtual environment to be activated multiple times and also every time \
    # the prompt is rendered. \
    && echo "unset BASH_ENV PROMPT_COMMAND ENV" >> ${APP_ROOT}/venv/bin/activate

# activate virtualenv with workaround RHEL/CentOS 8+
ENV BASH_ENV="${APP_ROOT}/venv/bin/activate" \
    ENV="${APP_ROOT}/venv/bin/activate" \
    PROMPT_COMMAND=". ${APP_ROOT}/venv/bin/activate" \
    PATH="${APP_ROOT}/venv/bin:${PATH}"

# Run the application
EXPOSE 8080
CMD ["uvicorn", "ols:app", "--host", "0.0.0.0", "--port", "8080"]

LABEL io.k8s.display-name="OpenShift LightSpeed Service" \
      io.k8s.description="AI-powered OpenShift Assistant Service." \
      io.openshift.tags="openshift-lightspeed,ols"