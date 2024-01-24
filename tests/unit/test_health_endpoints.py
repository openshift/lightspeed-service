"""Unit tests for health REST API endpoints handlers."""

from ols.app.endpoints.health import liveness_probe, readiness_probe


def test_readiness_probe():
    """Test the readiness_probe function."""
    # the tested function returns constant right now
    # i.e. it does not depend on application state
    response = readiness_probe()
    assert response == {"status": "1"}


def test_liveness_probe():
    """Test the liveness_probe function."""
    # the tested function returns constant right now
    # i.e. it does not depend on application state
    response = liveness_probe()
    assert response == {"status": "1"}
