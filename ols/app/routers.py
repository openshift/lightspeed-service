"""REST API routers."""

from fastapi import FastAPI

from ols.app.endpoints import authorized, feedback, health, hitl, ols, streaming_ols
from ols.app.metrics import metrics


def include_routers(app: FastAPI) -> None:
    """Include FastAPI routers for different endpoints.

    Args:
        app: The `FastAPI` app instance.
    """
    app.include_router(ols.router, prefix="/v1")
    app.include_router(streaming_ols.router, prefix="/v1")
    app.include_router(feedback.router, prefix="/v1")
    app.include_router(hitl.router, prefix="/v1/hitl")
    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(authorized.router)
