"""REST API routers."""

from fastapi import FastAPI

from ols.app import metrics
from ols.app.endpoints import feedback, health, ols


def include_routers(app: FastAPI) -> None:
    """Include FastAPI routers for different endpoints.

    Args:
        app: The `FastAPI` app instance.
    """
    app.include_router(ols.router, prefix="/v1")
    app.include_router(feedback.router, prefix="/v1")
    app.include_router(health.router)
    app.mount("/metrics", metrics.metrics_app)
