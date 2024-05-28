"""Unit tests for health endpoints handlers."""

import pytest

from ols import config
from ols.app.endpoints.health import (
    liveness_probe_get_method,
    liveness_probe_head_method,
    readiness_probe_get_method,
)
from ols.app.models.config import ConversationCacheConfig
from ols.app.models.models import HealthResponse, ReadinessResponse


@pytest.fixture(scope="function")
def _config_with_conversation_cache():
    """Fixture to set up the config with conversation cache."""
    # NOTE: The `conversation_cache` in config is a (cached) instance of
    # `Cache` class returned by the `CacheFactory`.
    # The cache is not used in the current test, so we can just return True.
    config.ols_config.conversation_cache = ConversationCacheConfig(
        {"type": "memory", "memory": {"max_size": 100}}
    )


def test_readiness_probe_get_method(_config_with_conversation_cache):
    """Test the readiness_probe function."""
    # the tested function returns constant right now
    # i.e. it does not depend on application state
    response = readiness_probe_get_method()
    assert response == ReadinessResponse(ready="True", reason="service is ready")


def test_liveness_probe_get_method():
    """Test the liveness_probe function."""
    # the tested function returns constant right now
    # i.e. it does not depend on application state
    response = liveness_probe_get_method()
    assert response == HealthResponse(status={"status": "healthy"})


def test_liveness_probe_head_method():
    """Test the liveness_probe function."""
    # the tested function returns constant right now
    # i.e. it does not depend on application state
    response = liveness_probe_head_method()
    assert response is None
