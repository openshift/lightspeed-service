"""Unit tests for feedback endpoint handlers."""

import json
from unittest.mock import patch

import pytest

from ols.app.endpoints import feedback
from ols.app.models.config import UserDataCollection
from ols.utils.config import ConfigManager


@pytest.fixture
def feedback_location(tmpdir):
    """Fixture sets feedback location to tmpdir and return the path."""
    config_manager = ConfigManager()
    config_manager.init_empty_config()
    full_path = (tmpdir / "feedback").strpath
    config_manager.get_ols_config().user_data_collection = UserDataCollection(
        feedback_disabled=False, feedback_storage=full_path
    )
    return full_path, config_manager


def store_fake_feedback(path, filename, data):
    """Store feedback data."""
    with open(f"{path}/{filename}.json", "w") as f:
        f.write(json.dumps(data))


def load_fake_feedback(filename):
    """Load feedback data."""
    config_manager = ConfigManager()
    ols_config = config_manager.get_ols_config()
    feedback_file = (
        f"{ols_config.user_data_collection.feedback_storage}/{filename}.json"
    )
    stored_data = json.loads(open(feedback_file).read())
    return stored_data


def test_get_feedback_status(feedback_location):
    """Test is_feedback_enabled function."""
    _, config_manager = feedback_location
    config_manager.get_ols_config().user_data_collection.feedback_disabled = False
    assert feedback.is_feedback_enabled()

    config_manager.get_ols_config().user_data_collection.feedback_disabled = True
    assert not feedback.is_feedback_enabled()


def test_store_feedback(feedback_location):
    """Test store_feedback function."""
    user_id = "12345678-abcd-0000-0123-456789abcdef"
    feedback_data = {"testy": "test"}
    with patch("ols.app.endpoints.feedback.get_suid", return_value="fake-uuid"):
        feedback.store_feedback(user_id, feedback_data)

    stored_data = load_fake_feedback("fake-uuid")
    assert stored_data == {
        "user_id": "12345678-abcd-0000-0123-456789abcdef",
        **feedback_data,
    }
