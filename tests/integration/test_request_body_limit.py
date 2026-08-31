"""Integration tests for the request body size limit middleware."""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

import logging

import pytest
from fastapi.testclient import TestClient

from ols import config
from ols.constants import MAX_REQUEST_BODY_SIZE


@pytest.fixture(scope="function")
def _setup():
    """Set up the test client with debug-level logging enabled."""
    config.reload_from_yaml_file("tests/config/config_for_integration_tests.yaml")

    # app.main needs to be imported after the configuration is read
    from ols.app.main import app  # pylint: disable=C0415

    config.dev_config.disable_auth = True
    pytest.client = TestClient(app)


def test_oversized_body_rejected_before_debug_logging(_setup, caplog):
    """Oversized bodies get a 413 and are never buffered by the debug logger.

    The limiter must be the outermost middleware so it wraps ``receive``
    before ``log_requests_responses`` reads the body. If it were inner, the
    debug logger would buffer and log the whole oversized payload first.
    """
    caplog.set_level(logging.DEBUG, logger="ols.app.main")

    marker = "OVERSIZE_BODY_MARKER"
    body = marker.encode() + b"A" * (MAX_REQUEST_BODY_SIZE + 1)

    response = pytest.client.post("/v1/query", content=body)

    assert response.status_code == 413
    # The oversized body must never reach the debug log: the limiter rejects
    # it before log_requests_responses can buffer and decode it.
    assert marker not in caplog.text


def test_within_limit_body_is_logged(_setup, caplog):
    """Control: a small body passes the limiter and is logged at debug level.

    Proves the debug logger does buffer and log request bodies, so the
    oversized-body assertion above is meaningful rather than vacuous.
    """
    caplog.set_level(logging.DEBUG, logger="ols.app.main")

    marker = "SMALL_BODY_MARKER"

    response = pytest.client.post("/v1/query", content=marker.encode())

    assert response.status_code != 413
    assert marker in caplog.text
