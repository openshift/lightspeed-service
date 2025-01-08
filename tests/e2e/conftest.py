"""Additional arguments for pytest."""

import json
import logging
import os
import tarfile
import tempfile
import uuid

import pytest
from pytest import TestReport
from reportportal_client import RPLogger

from scripts.upload_artifact_s3 import upload_artifact_s3
from tests.e2e import test_api

aws_env: dict[str, str] = {}

# this flag is set to True when synthetic test report was already generated
pytest.makereport_called = False


def pytest_runtest_makereport(item, call) -> TestReport:
    """Generate a synthetic test report.

    If OLS did not come up in time generate a synethic test entry,
    generate a normal test report entry otherwise.
    """
    # The first time we try to generate a test report, check if OLS timed out on startup
    # if so, generate a synthetic test report for the wait_for_ols timeout.
    if not test_api.OLS_READY and not pytest.makereport_called:
        pytest.makereport_called = True
        return TestReport(
            "test_wait_for_ols",
            ("", 0, ""),
            {},
            "failed",
            "wait for OLS to startup before running tests",
            "call",
            [],
        )
    # The second time we are called to generate a report, assuming OLS timed out,
    # exit pytest so we don't try to run any more tests (they will fail anyway since
    # OLS didn't come up)
    # There is no clean way to return the synthetic test report above *and* exit pytest
    # in a single invocation
    if not test_api.OLS_READY:
        pytest.exit("OLS did not become ready!", 1)
    # If OLS did come up cleanly during setup, then just generate normal test reports for all tests
    return TestReport.from_item_and_call(item, call)


def pytest_addoption(parser):
    """Argument parser for pytest."""
    parser.addoption(
        "--eval_provider",
        default="watsonx",
        type=str,
        help="Provider name, currently used only to form output file name.",
    )
    parser.addoption(
        "--eval_model",
        default="ibm/granite-3-8b-instruct",
        type=str,
        help="Model for which responses will be evaluated.",
    )
    parser.addoption(
        "--eval_provider_model_id",
        nargs="+",
        default=[
            # "bam+ibm/granite-3-8b-instruct",
            "watsonx+ibm/granite-3-8b-instruct",
            "openai+gpt-4o-mini",
            "azure_openai+gpt-4o-mini",
        ],
        type=str,
        help="Identifier for Provider/Model to be used for model eval.",
    )
    parser.addoption(
        "--eval_out_dir",
        default=None,
        type=str,
        help="Result destination.",
    )
    parser.addoption(
        "--eval_query_ids",
        nargs="+",
        default=None,
        help="Ids of questions to be validated. Check json file for valid ids.",
    )
    parser.addoption(
        "--eval_scenario",
        choices=["with_rag", "without_rag"],
        default="with_rag",
        type=str,
        help="Scenario for which responses will be evaluated.",
    )
    parser.addoption(
        "--qna_pool_file",
        default=None,
        type=str,
        help="Additional file having QnA pool in parquet format.",
    )
    parser.addoption(
        "--eval_type",
        choices=["consistency", "model", "all"],
        default="model",
        help="Evaluation type.",
    )
    parser.addoption(
        "--eval_metrics",
        nargs="+",
        default=["cos_score"],
        help="Evaluation score/metric.",
    )
    parser.addoption(
        "--eval_modes",
        nargs="+",
        default=["ols"],
        help="Evaluation modes ex: with just prompt/rag etc.",
    )
    parser.addoption(
        "--rp_name",
        action="store",
        default="e2e-ols-cluster",
        help="Enable report portal upload",
    )


@pytest.fixture(scope="session")
def rp_logger():
    """Set up logging for report portal.

    Returns: logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.setLoggerClass(RPLogger)
    return logger


def write_json_to_temp_file(json_data):
    """Write json to a temporary file.

    Args:
        json_data (dict): dict containing configuration from pytest.ini
    Returns:
        temporary file's name
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        json.dump(json_data, temp_file)
        temp_file.flush()
        return temp_file.name


def create_datarouter_config_file(session):
    """Create datarouter config file."""
    project = session.config.getini("rp_project")
    assert (
        project is not None
    ), "create_datarouter_config_file: 'rp_project' attribute not found in session.config"

    endpoint = session.config.getini("rp_endpoint")
    assert (
        endpoint is not None
    ), "create_datarouter_config_file: 'rp_endpoint' attribute not found in session.config"

    endpoint = endpoint.replace("https://", "")
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


def add_secret_to_env(env) -> None:
    """Add aws secrets from environment variable to dict.

    Args:
        env (env): environment variable name
    Returns:
        Null
    """
    name = env[:-5]
    with open(os.environ[env], encoding="utf-8") as file:
        content = file.read()
        aws_env[name] = content


def get_secret_value(env: str) -> str:
    """Handle secrets delivered in env variables."""
    with open(os.environ[env], encoding="utf-8") as file:
        return file.read()


def pytest_sessionfinish(session):
    """Create datarouter compatible archive to upload into report portal."""
    # Sending reports to report portal
    try:
        datarouter_config = create_datarouter_config_file(session)
        archive_path = os.path.join(os.getcwd(), f"reportportal-{uuid.uuid4()}.tar.gz")
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(datarouter_config, arcname="data_router.json")
            file_path = os.environ["ARTIFACT_DIR"]
            for file in os.listdir(file_path):
                if file.endswith(".xml"):
                    print(f"Found xml to add in archive {file}.")
                    tar.add(
                        os.path.join(file_path, file),
                        arcname=os.path.join("data", "results", file),
                    )
        print(f"Saved Report Portal datarouter archive to {archive_path}.")
    except Exception as e:
        print(f"Error creating RP archive: {e}")
        return
    try:
        add_secret_to_env("AWS_ACCESS_KEY_ID_PATH")
        add_secret_to_env("AWS_BUCKET_PATH")
        add_secret_to_env("AWS_REGION_PATH")
        add_secret_to_env("AWS_SECRET_ACCESS_KEY_PATH")
        upload_artifact_s3(aws_env=aws_env)
    except KeyError:
        print(
            "Could not find aws credentials to upload to S3. "
            "Skipping reporting to Report portal."
        )
