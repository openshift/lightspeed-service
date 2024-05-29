"""Additional arguments for pytest."""
import json
import uuid
import tempfile
import os
import logging
import tarfile
from typing import Dict
import subprocess
import pytest
from reportportal_client import RPLogger

aws_env: Dict[str, str] = {}


@pytest.fixture(scope="session")
def rp_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.setLoggerClass(RPLogger)
    return logger


def pytest_addoption(parser):
    """Argument parser for pytest."""
    parser.addoption(
        "--eval_model",
        default="gpt",
        type=str,
        help="Model to be evaluated.",
    )
    parser.addoption(
        "--rp_enabled", action="store_true", default=False, help="Enable report portal upload"
    )
    parser.addoption(
        "--rp_name",
        action="store",
        default="e2e-ols-cluster",
        help="Enable report portal upload",
    )


def write_json_to_temp_file(json_data):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        json.dump(json_data, temp_file)
        temp_file.flush()
        return temp_file.name


def create_datarouter_config_file(session):
    """Creates datarouter config file."""
    project = session.config.getini("rp_project")
    endpoint = session.config.getini("rp_endpoint").replace("https://", "")
    launch = session.config.option.rp_name
    launch_desc = session.config.getini("rp_launch_description") or ""
    json_data = {
        "targets": {
            "reportportal": {
                "config": {
                    "hostname": endpoint,
                    "project": project,
                },
                "processing": {
                    "apply_tfa": True,
                    "property_filter": ["^(?!(polarion|iqe_blocker).*$).*"],
                    "launch": {
                        "name": launch,
                        "description": launch_desc,
                    },
                },
            }
        }
    }

    temp_filename = write_json_to_temp_file(json_data)
    return temp_filename


def upload_artifact_s3():
    """Runs upload-artifact-s3 tool.

    aws_env (dict): Dictionary with AWS secrets, must contain the following keys:   AWS_REGION
    AWS_BUCKET   AWS_ACCESS_KEY_ID   AWS_SECRET_ACCESS_KEY
    """
    try:
        retcode = subprocess.run(["upload-artifact-s3"], env=aws_env).returncode
        if retcode != 0:
            logging.info("Failed to upload artifacts to S3")
        logging.info("Uploaded archive to S3")

    except Exception as e:
        logging.info(f"Error uploading archive to S3: {e}")


def add_secret_to_env(env) -> None:
    name = env[:-5]
    with open(os.environ[env]) as file:
        content = file.read()
        aws_env[name] = content
    return


def get_secret_value(env: str) -> str:
    """Handles secrets delivered in env variables."""
    with open(os.environ[env]) as file:
        return file.read()


def pytest_sessionfinish(session):
    """Creates datarouter compatible archive to upload into report portal."""

    # Sending reports to report portal
    if session.config.option.rp_enabled:
        try:
            datarouter_config = create_datarouter_config_file(session)
            archive_path = os.path.join(os.getcwd(), f"reportportal-{uuid.uuid4()}.tar.gz")
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(datarouter_config, arcname="data_router.json")
                for file in os.listdir(os.environ['ARTIFACT_DIR']):
                    if file.endswith(".xml"):
                        logging.info(f"Found xml to add in archive {file}.")
                        tar.add(file, arcname=os.path.join("data", "results", file))
            logging.info(f"Saved Report Portal datarouter archive to {archive_path}.")
        except Exception as e:
            logging.info(f"Error creating RP archive: {e}")
            return None
        try:
            add_secret_to_env("AWS_ACCESS_KEY_ID_PATH")
            add_secret_to_env("AWS_BUCKET_PATH")
            add_secret_to_env("AWS_REGION_PATH")
            add_secret_to_env("AWS_SECRET_ACCESS_KEY_PATH")
            upload_artifact_s3()
        except KeyError:
            logging.info(
                "Could not find aws credentials to upload to S3. "
                "Skipping reporting to Report portal."
            )

        return None