"""Unit tests for runner module - specifically the store_config function."""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from runner import store_config


@pytest.fixture
def config_location(tmpdir):
    """Fixture provides a temporary config storage location."""
    return (tmpdir / "config").strpath


@pytest.fixture
def mock_logger():
    """Fixture provides a mock logger for testing."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def mock_config(config_location):
    """Fixture provides a mock config object for testing."""
    mock_config = MagicMock()
    mock_config.ols_config.user_data_collection.config_storage = config_location
    return mock_config


@pytest.fixture
def sample_config_file():
    """Create a temporary config file with sample content."""
    config_content = """# Sample configuration
ols_config:
  default_provider: test_provider
  default_model: test_model
  user_data_collection:
    feedback_disabled: true
    transcripts_disabled: true
    config_disabled: false
    config_storage: "/tmp/config"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        f.flush()
        yield f.name
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


def test_store_config_enabled(
    config_location, sample_config_file, mock_logger, mock_config
):
    """Test that config is stored when enabled."""
    store_config(sample_config_file, mock_logger, mock_config)

    # Verify that a config file was created
    config_files = list(Path(config_location).glob("*.json"))
    assert len(config_files) == 1, f"Expected 1 config file, found {len(config_files)}"

    # Verify the content
    with open(config_files[0], "r") as f:
        stored_data = json.load(f)

    assert "metadata" in stored_data
    assert "configuration" in stored_data
    assert "timestamp" in stored_data["metadata"]
    assert "service_version" in stored_data["metadata"]
    assert "config_file_path" in stored_data["metadata"]
    assert "backend" in stored_data["metadata"]

    assert stored_data["metadata"]["config_file_path"] == sample_config_file
    assert stored_data["metadata"]["backend"] == "lightspeed-service"
    assert "ols_config:" in stored_data["configuration"]
    assert "test_provider" in stored_data["configuration"]


def test_store_config_creates_directory(tmpdir, sample_config_file, mock_logger):
    """Test that config storage creates directory if it doesn't exist."""
    # Use a non-existent nested directory path
    nested_path = tmpdir / "nested" / "config" / "storage"
    full_path = nested_path.strpath

    # Create a mock config with the nested path
    mock_config = MagicMock()
    mock_config.ols_config.user_data_collection.config_storage = full_path

    # Directory shouldn't exist initially
    assert not Path(full_path).exists()

    # Call store_config
    store_config(sample_config_file, mock_logger, mock_config)

    # Directory should be created
    assert Path(full_path).exists()
    assert Path(full_path).is_dir()

    # Config file should be stored
    config_files = list(Path(full_path).glob("*.json"))
    assert len(config_files) == 1


def test_store_config_unique_filenames(
    config_location, sample_config_file, mock_logger, mock_config
):
    """Test that multiple calls create files with unique names."""
    # Call store_config multiple times
    store_config(sample_config_file, mock_logger, mock_config)
    store_config(sample_config_file, mock_logger, mock_config)
    store_config(sample_config_file, mock_logger, mock_config)

    # Should have 3 unique files
    config_files = list(Path(config_location).glob("*.json"))
    assert len(config_files) == 3

    # All filenames should be unique
    filenames = [f.name for f in config_files]
    assert len(set(filenames)) == 3


@patch("runner.__version__", "1.2.3-test")
def test_store_config_includes_version(
    config_location, sample_config_file, mock_logger, mock_config
):
    """Test that stored config includes the service version."""
    store_config(sample_config_file, mock_logger, mock_config)

    config_files = list(Path(config_location).glob("*.json"))
    with open(config_files[0], "r") as f:
        stored_data = json.load(f)

    assert stored_data["metadata"]["service_version"] == "1.2.3-test"


def test_store_config_preserves_yaml_content(
    config_location, sample_config_file, mock_logger, mock_config
):
    """Test that original YAML content is preserved exactly."""
    # Read the original content
    with open(sample_config_file, "r") as f:
        original_content = f.read()

    store_config(sample_config_file, mock_logger, mock_config)

    config_files = list(Path(config_location).glob("*.json"))
    with open(config_files[0], "r") as f:
        stored_data = json.load(f)

    # The stored configuration should match the original exactly
    assert stored_data["configuration"] == original_content


def test_store_config_json_format(
    config_location, sample_config_file, mock_logger, mock_config
):
    """Test that stored file is valid JSON with proper structure."""
    store_config(sample_config_file, mock_logger, mock_config)

    config_files = list(Path(config_location).glob("*.json"))

    # Should be valid JSON
    with open(config_files[0], "r") as f:
        stored_data = json.load(f)  # This will raise if invalid JSON

    # Should have expected structure
    expected_keys = {"metadata", "configuration"}
    assert set(stored_data.keys()) == expected_keys

    expected_metadata_keys = {
        "timestamp",
        "service_version",
        "config_file_path",
        "backend",
    }
    assert set(stored_data["metadata"].keys()) == expected_metadata_keys

    # Configuration should be a string (YAML content)
    assert isinstance(stored_data["configuration"], str)
