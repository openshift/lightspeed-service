"""Handlers for OLS health REST API endpoints."""

from fastapi import APIRouter


router = APIRouter(tags=["health"])

# Still to be decided on their functionality


@router.post("/healthz")
@router.post("/readyz")
def health():
    """Health status of service."""
    return {"status": "1"}
