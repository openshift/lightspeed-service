"""Entry point to FastAPI-based web service."""

import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

from ols.app import metrics, routers
from ols.src.ui.gradio_ui import GradioUI
from ols.utils import config

app = FastAPI(
    title="Swagger OpenShift LightSpeed Service - OpenAPI",
    description="""OpenShift LightSpeed Service API specification.""",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


logger = logging.getLogger(__name__)

if config.dev_config.enable_dev_ui:
    app = GradioUI().mount_ui(app)
else:
    logger.info(
        "Embedded Gradio UI is disabled. To enable set `enable_dev_ui: true` "
        "in the `dev_config` section of the configuration file."
    )


# update provider and model as soon as possible so the metrics will be visible
# even for first scraping
metrics.setup_model_metrics(config)


@app.middleware("")
async def rest_api_counter(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Middleware with REST API counter update logic."""
    path = request.url.path

    # measure time to handle duration + update histogram
    with metrics.response_duration_seconds.labels(path).time():
        response = await call_next(request)

    # ignore /metrics endpoint that will be called periodically
    if not path.endswith("/metrics"):
        # just update metrics
        metrics.rest_api_calls_total.labels(path, response.status_code).inc()
    return response


routers.include_routers(app)
