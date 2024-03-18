"""Unit tests for feedback endpoint handlers."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ols.app.endpoints import feedback
from ols.app.models.config import UserDataCollection
from ols.utils import config


@pytest.fixture
def feedback_location(tmpdir):
    """Fixture sets feedback location to tmpdir and return the path."""
    config.init_empty_config()
    config.ols_config.user_data_collection = UserDataCollection(
        feedback_disabled=False, feedback_storage=tmpdir.strpath
    )
    return tmpdir.strpath


def store_fake_feedback(path, filename, data):
    """Store feedback data."""
    with open(f"{path}/{filename}.json", "w") as f:
        f.write(json.dumps(data))


def load_fake_feedback(filename):
    """Load feedback data."""
    feedback_file = (
        f"{config.ols_config.user_data_collection.feedback_storage}/{filename}.json"
    )
    stored_data = json.loads(open(feedback_file).read())
    return stored_data


def test_get_feedback_status(feedback_location):
    """Test get_feedback_status function."""
    config.ols_config.user_data_collection.feedback_disabled = False
    assert feedback.get_feedback_status()

    config.ols_config.user_data_collection.feedback_disabled = True
    assert not feedback.get_feedback_status()


def test_list_feedbacks(feedback_location):
    """Test list_feedbacks function."""
    feedbacks = feedback.list_feedbacks()
    assert feedbacks == []

    store_fake_feedback(feedback_location, "test", {"some": "data"})
    feedbacks = feedback.list_feedbacks()
    assert feedbacks == ["test"]


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


def test_remove_feedback(feedback_location):
    """Test remove_feedback function."""
    feedback_id = "fake-id"
    store_fake_feedback(feedback_location, feedback_id, {"some": "data"})
    assert len(list(Path(feedback_location).iterdir())) == 1

    feedback.remove_feedback(feedback_id)

    assert len(list(Path(feedback_location).iterdir())) == 0
