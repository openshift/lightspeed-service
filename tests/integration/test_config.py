"""Integration tests for config loader."""

import pytest

from ols import config


def test_load_non_existent_config():
    """Test how loading of non-existent config is handled."""
    with pytest.raises(
        FileNotFoundError, match="tests/config/non_existent_config.yaml"
    ):
        config.reload_from_yaml_file("tests/config/non_existent_config.yaml")
