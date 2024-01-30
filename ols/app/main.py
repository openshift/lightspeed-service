"""Entry point to FastAPI-based web service."""

import logging
import os

from fastapi import FastAPI

from ols.app.endpoints import feedback, health, ols
from ols.src.ui.gradio_ui import GradioUI
from ols.utils import config
from ols.utils.logging import configure_logging

app = FastAPI(
    title="Swagger OpenShift LightSpeed Service - OpenAPI",
    description="""OpenShift LightSpeed Service API specification.""",
)


config.init_config(os.environ.get("OLS_CONFIG_FILE", "olsconfig.yaml"))


configure_logging(config.ols_config.logging_config)
logger = logging.getLogger(__name__)


if config.dev_config.enable_dev_ui:
    app = GradioUI().mount_ui(app)
else:
    logger.info(
        "Embedded Gradio UI is disabled. To enable set enable_dev_ui: true "
        "in the dev section of the configuration file"
    )


def include_routers(app: FastAPI):
    """Include FastAPI routers for different endpoints.

    Args:
        app: The `FastAPI` app instance.
    """
    app.include_router(ols.router, prefix="/v1")
    app.include_router(feedback.router, prefix="/v1")
    app.include_router(health.router)


include_routers(app)
