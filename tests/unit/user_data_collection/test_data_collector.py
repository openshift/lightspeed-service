"""Unit tests for the data_collector module."""

import tarfile
from unittest.mock import Mock, patch

from requests.models import Response

from ols.user_data_collection import data_collector
from ols.user_data_collection.data_collector import (
    collect_ols_data_from as orignal_collect_ols_data_from,
)


def test_collect_ols_data_from(tmp_path):
    """Test the collect_ols_data_from function."""
    with open(tmp_path / "root.json", "w") as f:  # should be ignored
        f.write("{}")
    unknown_dir = tmp_path / "nested"
    unknown_dir.mkdir()
    with open(unknown_dir / "nested.json", "w") as f:  # should be ignored
        f.write("{}")
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    with open(feedback_dir / "feedback.json", "w") as f:
        f.write("{}")
    with open(feedback_dir / "not_a_json", "w") as f:  # should be ignored
        f.write("{}")
    transcipts_dir = tmp_path / "transcripts"
    transcipts_dir.mkdir()
    with open(transcipts_dir / "transcript.json", "w") as f:
        f.write("{}")

    collected = data_collector.collect_ols_data_from(tmp_path)

    assert collected == [
        feedback_dir / "feedback.json",
        transcipts_dir / "transcript.json",
    ]


def test_package_files_into_tarball(tmp_path):
    """Test the package_files_into_tarball function."""
    with open(tmp_path / "some.json", "w") as f:
        f.write("{}")
    extra_dir = tmp_path / "extra_dir"
    extra_dir.mkdir()
    with open(extra_dir / "extra.json", "w") as f:
        f.write("{}")

    tarball = data_collector.package_files_into_tarball(
        [
            tmp_path / "some.json",
            extra_dir / "extra.json",
        ],
        path_to_strip=tmp_path.as_posix(),
    )

    with tarfile.open(fileobj=tarball, mode="r:gz") as tar:
        files = tar.getnames()
    # asserts that files in tarball are stored without the full path
    assert files == ["some.json", "extra_dir/extra.json", "openshift_lightspeed.json"]


def test_delete_data(tmp_path):
    """Test the delete_data function."""
    with open(tmp_path / "some.json", "w") as f:
        f.write("something")
    assert len(list(tmp_path.iterdir())) == 1

    data_collector.delete_data([tmp_path / "some.json"])

    assert len(list(tmp_path.iterdir())) == 0


def mock_collect_ols_data_from(data_path):
    """Mock collect_ols_data_from function."""
    # call the original function and get its result
    original_result = orignal_collect_ols_data_from(data_path)

    # create a new file
    with open(data_path / "new_file.json", "w") as f:
        f.write("this is a new file that should not be collected")

    # return the original result
    return original_result


def test_new_files_stays(tmp_path):
    """Test gather_ols_user_data.

    During the test, a new file is added to directory that is being
    watched for data collection. The new file should stay after the data
    collection - it is not in the collected files (and send), hence it
    shouldn't be removed.
    """

    def mock_response(*args, **kwargs):
        mock_response = Mock(spec=Response)
        mock_response.status_code = 202
        mock_response.json.return_value = {"request_id": "some-request-id"}
        return mock_response

    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    with open(feedback_dir / "feedback.json", "w") as f:
        f.write("{}")
    assert [f.name for f in tmp_path.rglob("*")] == ["feedback", "feedback.json"]

    with (
        patch(
            "ols.user_data_collection.data_collector.collect_ols_data_from",
            new=mock_collect_ols_data_from,
        ),
        patch(
            "ols.user_data_collection.data_collector.upload_data_to_ingress",
            return_value=mock_response(),
        ),
    ):
        data_collector.gather_ols_user_data(tmp_path)

    # feedback is a dir that is not removed and new_file.json is what
    # we are looking for here
    assert [f.name for f in tmp_path.rglob("*")] == ["new_file.json", "feedback"]
