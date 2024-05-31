"""Handlers for OLS health REST API endpoints.

These endpoints are used to check if service is live and prepared to accept
requests. Note that these endpoints can be accessed using GET or HEAD HTTP
methods. For HEAD HTTP method, just the HTTP response code is used.
"""

from fastapi import APIRouter

from ols.app.models.models import HealthResponse, LivenessResponse

router = APIRouter(tags=["health"])


# TODO: OLS-488 Define behaviour of /readiness and /liveness endpoints in OLS service
#
# Example status response:
# {
#     "status": "unhealthy",
#     "version": "1.0.0",
#     "dependencies": {
#         "database": "healthy",
#         "externalApi": "unhealthy"
#     }
# }


@router.get("/readiness")
def readiness_probe_get_method() -> HealthResponse:
    """Ready status of service."""
    return HealthResponse(status={"status": "healthy"})


@router.get("/liveness")
def liveness_probe_get_method() -> LivenessResponse:
    """Live status of service."""
    return LivenessResponse(alive=True)


@router.head("/readiness")
def readiness_probe_head_method() -> None:
    """Ready status of service."""
    return


@router.head("/liveness")
def liveness_probe_head_method() -> None:
    """Live status of service."""
    return
