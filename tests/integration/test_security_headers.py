"""Integration tests for security headers middleware."""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

import pytest
import requests
from fastapi.testclient import TestClient

from ols import config


@pytest.fixture(scope="function")
def _setup():
    """Setups the test client."""
    config.reload_from_yaml_file("tests/config/config_for_integration_tests.yaml")

    # app.main need to be imported after the configuration is read
    from ols.app.main import app  # pylint: disable=C0415

    pytest.client = TestClient(app)


def test_security_headers_present_on_api_endpoints(_setup):
    """Verify security headers are present on API endpoints."""
    response = pytest.client.get("/openapi.json")
    assert response.status_code == requests.codes.ok

    # X-Content-Type-Options should be present
    assert "x-content-type-options" in response.headers
    assert response.headers["x-content-type-options"] == "nosniff"


def test_security_headers_absent_on_metrics(_setup):
    """Verify security headers are NOT present on metrics endpoint."""
    response = pytest.client.get("/metrics")
    assert response.status_code == requests.codes.ok

    # Security headers should NOT be present on metrics
    assert "x-content-type-options" not in response.headers
    assert "strict-transport-security" not in response.headers


def test_security_headers_absent_on_readiness(_setup):
    """Verify security headers are NOT present on readiness endpoint."""
    response = pytest.client.get("/readiness")
    # Readiness may return 503 if dependencies aren't ready, that's OK for this test
    assert response.status_code in [
        requests.codes.ok,
        requests.codes.service_unavailable,
    ]

    # Security headers should NOT be present on readiness, regardless of status code
    assert "x-content-type-options" not in response.headers
    assert "strict-transport-security" not in response.headers


def test_security_headers_absent_on_liveness(_setup):
    """Verify security headers are NOT present on liveness endpoint."""
    response = pytest.client.get("/liveness")
    assert response.status_code == requests.codes.ok

    # Security headers should NOT be present on liveness
    assert "x-content-type-options" not in response.headers
    assert "strict-transport-security" not in response.headers


def test_hsts_only_when_tls_enabled(_setup):
    """Verify HSTS header is only added when TLS is enabled."""
    response = pytest.client.get("/openapi.json")
    assert response.status_code == requests.codes.ok

    # In test config, TLS is typically disabled
    # So HSTS should NOT be present
    if config.dev_config.disable_tls:
        assert "strict-transport-security" not in response.headers
    else:
        # If TLS is enabled, HSTS should be present
        assert "strict-transport-security" in response.headers
        assert "max-age=31536000" in response.headers["strict-transport-security"]
        assert "includeSubDomains" in response.headers["strict-transport-security"]
