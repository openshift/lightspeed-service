"""Unit tests for the data_collector module."""

import logging
import os
import pathlib
import tarfile
from unittest.mock import Mock, patch

import pytest
from requests.models import Response

from ols.utils import suid


@pytest.fixture(scope="module", autouse=True)
def with_mocked_config():
    """Run test with mocked config.

    We don't need that for unit tests.
    """
    global data_collector
    global original_collect_ols_data_from
    with patch("ols.config", new=Mock()):
        from ols.user_data_collection import data_collector
        from ols.user_data_collection.data_collector import (
            collect_ols_data_from as original_collect_ols_data_from,
        )

        yield


def mock_ingress_response(*args, **kwargs):
    """Mock the response from the Ingress."""
    mock_response = Mock(spec=Response)
    mock_response.status_code = 202
    mock_response.json.return_value = {"request_id": "some-request-id"}
    return mock_response


def create_file_with_size(filename, size):
    """Create a file with a specific size.

    Args:
        filename: Name of the file to be created.
        size: Size of the file in bytes.
    """
    with open(filename, "wb") as f:
        f.write(os.urandom(size))


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
    transcipts_dir = tmp_path / "transcripts" / "uuid" / "uuid"
    transcipts_dir.mkdir(parents=True)
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
    assert files == ["some.json", "extra_dir/extra.json"]


def test_delete_data(tmp_path):
    """Test the delete_data function."""
    with open(tmp_path / "some.json", "w") as f:
        f.write("something")
    assert len(list(tmp_path.iterdir())) == 1

    data_collector.delete_data([tmp_path / "some.json"])

    assert len(list(tmp_path.iterdir())) == 0


def mock_collect_ols_data_from(data_path: str) -> list[pathlib.Path]:
    """Mock collect_ols_data_from function."""
    # call the original function and get its result
    original_result = original_collect_ols_data_from(data_path)

    # create a new file
    with open(data_path + "/new_file.json", "w") as f:
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
            return_value=mock_ingress_response(),
        ),
    ):
        data_collector.gather_ols_user_data(tmp_path.as_posix())

    # feedback is a dir that is not removed and new_file.json is what
    # we are looking for here
    assert sorted([f.name for f in tmp_path.rglob("*")]) == [
        "feedback",
        "new_file.json",
    ]


def test_gather_ols_user_data_nothing_to_collect(tmp_path, caplog):
    """Test gather_ols_user_data with no data to collect."""
    caplog.set_level(logging.INFO)

    data_collector.gather_ols_user_data(tmp_path.as_posix())

    assert "contains no data, nothing to do..." in caplog.text


def test_gather_ols_user_data_full_flow(tmp_path, caplog):
    """Test gather_ols_user_data.

    This is basically script e2e test. We just mocks the ingress.
    """
    caplog.set_level(logging.DEBUG)
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    with open(feedback_dir / f"{suid.get_suid()}.json", "w") as f:
        f.write("{}")
    assert len(list(feedback_dir.iterdir())) == 1

    with patch(
        "ols.user_data_collection.data_collector.upload_data_to_ingress",
        return_value=mock_ingress_response(),
    ):
        data_collector.gather_ols_user_data(tmp_path.as_posix())

    # assert correct logs and empty dir where data was
    assert "collected 1 files (splitted to 1 chunks)" in caplog.text
    assert "uploaded data removed" in caplog.text
    assert len(list(feedback_dir.iterdir())) == 0


def test_chunk_data(tmpdir):
    """Test the chunk_data function."""
    files = [pathlib.Path(f"{tmpdir.strpath}/{i}.json") for i in range(3)]
    for file in files:
        create_file_with_size(file.as_posix(), 20)

    # file bigger than chunk size - file per chunk
    chunks = data_collector.chunk_data(files, 19)
    assert chunks == [[files[0]], [files[1]], [files[2]]]

    # file equal to chunk size - file per chunk
    chunks = data_collector.chunk_data(files, 20)
    assert chunks == [[files[0]], [files[1]], [files[2]]]

    # file smaller than chunk size, but chunk smaller than two files
    chunks = data_collector.chunk_data(files, 25)
    assert chunks == [[files[0]], [files[1]], [files[2]]]

    # bigger chunk - two files fit in a chunk
    chunks = data_collector.chunk_data(files, 45)
    assert chunks == [[files[0], files[1]], [files[2]]]

    # bigger chunk - everything fits into one
    chunks = data_collector.chunk_data(files, 100)
    assert chunks == [[files[0], files[1], files[2]]]


def test_ensure_data_is_not_bigger_than_defined(tmpdir):
    """Test the ensure_data_is_not_bigger_than_defined function."""
    feedback_dir = tmpdir / "feedback"
    feedback_dir.mkdir()
    files = [pathlib.Path(f"{feedback_dir.strpath}/{i}.json") for i in range(3)]
    for file in files:
        create_file_with_size(file.as_posix(), 20)

    assert len(feedback_dir.listdir()) == 3

    data_collector.ensure_data_dir_is_not_bigger_than_defined(tmpdir.strpath, 50)

    assert len(feedback_dir.listdir()) == 2


def test_access_token_from_offline_token():
    """Test the access_token_from_offline_token function."""
    with patch("requests.post", return_value=Response()):
        with pytest.raises(Exception, match="Response is not JSON"):
            data_collector.access_token_from_offline_token("offline_token")
