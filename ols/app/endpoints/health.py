"""Handlers for OLS health REST API endpoints.

These endpoints are used to check if service is live and prepared to accept
requests. Note that these endpoints can be accessed using GET or HEAD HTTP
methods. For HEAD HTTP method, just the HTTP response code is used.
"""

from fastapi import APIRouter

from ols import config
from ols.app.models.models import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


def cache_is_ready() -> bool:
    """Check if the cache is ready."""
    # NOTE: The `conversation_cache` in config is a (cached) instance of
    # `Cache` class returned by the `CacheFactory`.
    return config.conversation_cache.is_ready()


@router.get("/readiness")
def readiness_probe_get_method() -> ReadinessResponse:
    """Ready status of service."""
    if not cache_is_ready():
        return ReadinessResponse(ready=False, reason="cache is not ready")
    else:
        return ReadinessResponse(ready=True, reason="service is ready")


@router.get("/liveness")
def liveness_probe_get_method() -> HealthResponse:
    """Live status of service."""
    return HealthResponse(status={"status": "healthy"})


@router.head("/liveness")
def liveness_probe_head_method() -> None:
    """Live status of service."""
    return
