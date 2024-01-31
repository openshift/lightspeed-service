"""Handlers for OLS health REST API endpoints.

These endpoints are used to check if service is live and prepared to accept
requests. Note that these endpoints can be accessed using GET or HEAD HTTP
methods. For HEAD HTTP method, just the HTTP response code is used.
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])

# Still to be decided on their functionality


@router.get("/readiness")
def readiness_probe_get_method() -> dict[str, str]:
    """Ready status of service."""
    return {"status": "1"}


@router.get("/liveness")
def liveness_probe_get_method() -> dict[str, str]:
    """Live status of service."""
    return {"status": "1"}


@router.head("/readiness")
def readiness_probe_head_method() -> None:
    """Ready status of service."""
    return None


@router.head("/liveness")
def liveness_probe_head_method() -> None:
    """Live status of service."""
    return None
