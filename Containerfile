FROM registry.access.redhat.com/ubi9/ubi-minimal

ARG VERSION
ARG APP_ROOT=/app-root

RUN microdnf install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs \
    python3.11 python3.11-devel python3.11-pip jq shadow-utils \
    && microdnf clean all --enablerepo='*' \
    && useradd -r -u 1001 -g 0 -m -c "Default Application User" -d ${APP_ROOT} -s /bin/bash default

RUN pip3.11 install poetry

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
COPY . .
RUN chown -R 1001:0 ${APP_ROOT}

# default user for Python app
USER 1001

# envs related to correct venv activation
ENV APP_VENV=$APP_ROOT/.venv
ENV BASH_ENV=$APP_VENV/bin/activate
ENV ENV=$APP_VENV/bin/activate
ENV PROMPT_COMMAND=". $APP_VENV/bin/activate"
ENV PATH=$APP_VENV/bin:$PATH

# NOTE: build is a bit slow - would deserve further refactoring
# install the dependencies
RUN poetry config virtualenvs.create true \
  && poetry config virtualenvs.in-project true \
  && poetry config installer.max-workers 10 \
  && poetry install --no-dev --no-interaction --no-ansi

# run the application
EXPOSE 8080
CMD ["uvicorn", "lightspeed_service.main:app", "--host", "0.0.0.0", "--port", "8080"]

LABEL io.k8s.display-name="OpenShift LightSpeed Service" \
      io.k8s.description="AI-powered OpenShift Assistant Service." \
      io.openshift.tags="openshift-lightspeed,ols"
