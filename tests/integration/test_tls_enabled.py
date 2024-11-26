"""Integration tests for basic OLS REST API endpoints."""

from fastapi.testclient import TestClient

from ols import config


def test_setup_on_port_8443():
    """Setups the test client."""
    config.reload_from_yaml_file("tests/config/config_for_integration_tests_8443.yaml")

    # app.main need to be imported after the configuration is read
    from ols.app.main import app  # pylint: disable=C0415

    client = TestClient(app)
    assert client is not None
