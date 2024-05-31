"""Unit tests for health endpoints handlers."""

from ols import config
from ols.app.endpoints.health import (
    liveness_probe_get_method,
    liveness_probe_head_method,
    readiness_probe_get_method,
)
from ols.app.models.models import LivenessResponse, ReadinessResponse


def test_readiness_probe_get_method_index_is_ready():
    """Test the readiness_probe function when index is loaded."""
    # simulate that the index is loaded
    config._rag_index = True
    response = readiness_probe_get_method()
    assert response == ReadinessResponse(ready=True, reason="service is ready")

    # simulate that the index is not loaded, but it shouldn't as there
    # is no reference content in config
    config._rag_index = None
    config.ols_config.reference_content = None
    response = readiness_probe_get_method()
    assert response == ReadinessResponse(ready=True, reason="service is ready")


def test_readiness_probe_get_method_index_not_ready():
    """Test the readiness_probe function when index is not loaded."""
    # simulate that the index is not loaded
    config._rag_index = None
    config.ols_config.reference_content = "something else than None"

    response = readiness_probe_get_method()
    assert response == ReadinessResponse(ready=False, reason="index is not ready")


def test_liveness_probe_get_method():
    """Test the liveness_probe function."""
    # the tested function returns constant right now
    # i.e. it does not depend on application state
    response = liveness_probe_get_method()
    assert response == LivenessResponse(alive=True)


def test_liveness_probe_head_method():
    """Test the liveness_probe function."""
    # the tested function returns constant right now
    # i.e. it does not depend on application state
    response = liveness_probe_head_method()
    assert response is None
