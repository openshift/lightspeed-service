"""Handlers for OLS health REST API endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])

# Still to be decided on their functionality


@router.get("/readiness")
def readiness_probe():
    """Ready status of service."""
    return {"status": "1"}


@router.get("/liveness")
def liveness_probe():
    """Live status of service."""
    return {"status": "1"}
