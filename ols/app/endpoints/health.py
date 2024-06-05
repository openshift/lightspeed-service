"""Handlers for OLS health REST API endpoints.

These endpoints are used to check if service is live and prepared to accept
requests. Note that these endpoints can be accessed using GET or HEAD HTTP
methods. For HEAD HTTP method, just the HTTP response code is used.
"""

from fastapi import APIRouter

from ols import config
from ols.app.models.models import LivenessResponse, ReadinessResponse

router = APIRouter(tags=["health"])


def index_is_ready() -> bool:
    """Check if the index is loaded."""
    if config._rag_index is None and config.ols_config.reference_content is not None:
        return False
    else:
        return True


@router.get("/readiness")
def readiness_probe_get_method() -> ReadinessResponse:
    """Ready status of service."""
    if not index_is_ready():
        return ReadinessResponse(ready=False, reason="index is not ready")
    else:
        return ReadinessResponse(ready=True, reason="service is ready")


@router.get("/liveness")
def liveness_probe_get_method() -> LivenessResponse:
    """Live status of service."""
    return LivenessResponse(alive=True)
