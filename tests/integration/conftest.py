"""Configuration for integration tests."""

import pytest

from ols import config


@pytest.fixture(scope="function", autouse=True)
def ensure_empty_config_for_each_integration_test_by_default():
    """Set up fixture for all integration tests."""
    config.reload_empty()
