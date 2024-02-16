"""Entry point to FastAPI-based web service."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request

from ols.app import metrics, routers
from ols.src.ui.gradio_ui import GradioUI
from ols.utils import config
from ols.utils.logging import configure_logging

app = FastAPI(
    title="Swagger OpenShift LightSpeed Service - OpenAPI",
    description="""OpenShift LightSpeed Service API specification.""",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


cfg_file = os.environ.get("OLS_CONFIG_FILE", "olsconfig.yaml")
config.init_config(cfg_file)

configure_logging(config.ols_config.logging_config)
logger = logging.getLogger(__name__)
logger.info(f"Config loaded from {Path(cfg_file).resolve()}")


if config.dev_config.enable_dev_ui:
    app = GradioUI().mount_ui(app)
else:
    logger.info(
        "Embedded Gradio UI is disabled. To enable set enable_dev_ui: true "
        "in the dev section of the configuration file"
    )


@app.middleware("")
async def rest_api_counter(request: Request, call_next):
    """Middleware with REST API counter update logic."""
    # ignore /metrics endpoint that will be called periodically
    path = request.url.path
    if not path.endswith("/metrics/"):
        # just update metrics
        metrics.rest_api_calls.inc()
    return await call_next(request)


routers.include_routers(app)
