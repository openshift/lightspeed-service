"""Unit tests for feedback endpoint handlers."""

import json
from datetime import datetime
from unittest.mock import patch

import pytest

from ols import config
from ols.app.endpoints import feedback
from ols.app.models.config import UserDataCollection


@pytest.fixture
def feedback_location(tmpdir):
    """Fixture sets feedback location to tmpdir and return the path."""
    full_path = (tmpdir / "feedback").strpath
    config.ols_config.user_data_collection = UserDataCollection(
        feedback_disabled=False, feedback_storage=full_path
    )
    return full_path


def store_fake_feedback(path, filename, data):
    """Store feedback data."""
    with open(f"{path}/{filename}.json", "w") as f:
        f.write(json.dumps(data))


def load_fake_feedback(filename):
    """Load feedback data."""
    feedback_file = (
        f"{config.ols_config.user_data_collection.feedback_storage}/{filename}.json"
    )
    return json.loads(open(feedback_file).read())


def test_get_feedback_status(feedback_location):
    """Test is_feedback_enabled function."""
    config.ols_config.user_data_collection.feedback_disabled = False
    assert feedback.is_feedback_enabled()

    config.ols_config.user_data_collection.feedback_disabled = True
    assert not feedback.is_feedback_enabled()


@patch("ols.app.endpoints.feedback.datetime")
def test_store_feedback(mocked_datetime, feedback_location):
    """Test store_feedback function."""
    user_id = "12345678-abcd-0000-0123-456789abcdef"
    feedback_data = {"testy": "test"}

    mocked_datetime.utcnow = lambda: datetime(2000, 1, 1, 1, 23, 45)
    with patch("ols.app.endpoints.feedback.get_suid", return_value="fake-uuid"):
        feedback.store_feedback(user_id, feedback_data)

    stored_data = load_fake_feedback("fake-uuid")
    assert stored_data == {
        "user_id": "12345678-abcd-0000-0123-456789abcdef",
        "timestamp": "2000-01-01 01:23:45",
        **feedback_data,
    }
