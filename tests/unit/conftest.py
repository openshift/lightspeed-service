"""Configuration for unit tests."""

import pytest

from ols import config


@pytest.fixture(scope="function", autouse=True)
def ensure_empty_config_for_each_unit_test_by_default():
    """Set up fixture for all unit tests."""
    config.reload_empty()
