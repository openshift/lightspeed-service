"""Unit tests for health endpoints handlers."""

from ols.app.endpoints.health import (
    liveness_probe_get_method,
    liveness_probe_head_method,
    readiness_probe_get_method,
    readiness_probe_head_method,
)
from ols.app.models.models import HealthResponse, LivenessResponse


def test_readiness_probe_get_method():
    """Test the readiness_probe function."""
    # the tested function returns constant right now
    # i.e. it does not depend on application state
    response = readiness_probe_get_method()
    assert response == HealthResponse(status={"status": "healthy"})


def test_liveness_probe_get_method():
    """Test the liveness_probe function."""
    # the tested function returns constant right now
    # i.e. it does not depend on application state
    response = liveness_probe_get_method()
    assert response == LivenessResponse(alive=True)


def test_readiness_probe_head_method():
    """Test the readiness_probe function."""
    # the tested function returns constant right now
    # i.e. it does not depend on application state
    response = readiness_probe_head_method()
    assert response is None


def test_liveness_probe_head_method():
    """Test the liveness_probe function."""
    # the tested function returns constant right now
    # i.e. it does not depend on application state
    response = liveness_probe_head_method()
    assert response is None
