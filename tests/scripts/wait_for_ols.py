"""This module contains the wait_for_ols function and generate_junit_report function.

to check the readiness of the OLS and generate a JUnit report.
"""

import os
import sys
import time
import warnings
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import requests
from requests.exceptions import SSLError
from urllib3.exceptions import InsecureRequestWarning

from tests.scripts.must_gather import must_gather

sys.path.append(Path(__file__).resolve().parents[2].as_posix())

warnings.filterwarnings("ignore", category=InsecureRequestWarning)


# ruff: noqa: S501
def wait_for_ols(url, timeout=600, interval=10):
    """Wait for the OLS to become ready by checking its readiness endpoint.

    Args:
        url (str): The base URL of the OLS service.
        timeout (int, optional): The maximum time to wait in seconds. Default is 600.
        interval (int, optional): The interval between readiness checks in seconds. Default is 10.

    Returns:
        bool: True if OLS becomes ready within the timeout, False otherwise.
    """
    print("Starting wait_for_ols")
    attempts = int(timeout / interval)
    for attempt in range(1, attempts + 1):
        print(f"Checking OLS readiness, attempt {attempt} of {attempts}")
        try:
            response = requests.get(f"{url}/readiness", verify=True, timeout=5)
            if response.status_code == requests.codes.ok:
                print("OLS is ready")
                return True
        except SSLError:
            print("SSL error detected, retrying without SSL verification")
            try:
                response = requests.get(f"{url}/readiness", verify=False, timeout=5)
                if response.status_code == requests.codes.ok:
                    print("OLS is ready")
                    return True
            except requests.RequestException:
                pass
        except requests.RequestException:
            pass
        time.sleep(interval)
    print("Timed out waiting for OLS to become available")
    return False


def generate_junit_report(suite_id, success):
    """Generate a JUnit XML report indicating the success or failure of waiting for OLS.

    Args:
        suite_id (str): The ID of the test suite.
        success (bool): True if OLS became ready, False otherwise.
    """
    testsuite = ElementTree.Element(
        "testsuite", name=suite_id, tests="1", failures="0" if success else "1"
    )
    testcase = ElementTree.SubElement(
        testsuite, "testcase", name="wait_for_ols", classname=f"{suite_id}.wait_for_ols"
    )
    if not success:
        failure = ElementTree.SubElement(
            testcase, "failure", message="OLS failed to start up"
        )
        failure.text = "OLS did not become available in time"
    tree = ElementTree.ElementTree(testsuite)
    artifact_dir = os.getenv("ARTIFACT_DIR", ".")
    tree.write(f"{artifact_dir}/junit_setup_{suite_id}.xml")


if __name__ == "__main__":
    suite_id = os.getenv("SUITE_ID")
    ols_url = os.getenv("OLS_URL")
    if not ols_url:
        print("OLS_URL environment variable is not set")
        generate_junit_report(suite_id, False)
        exit(1)

    success = wait_for_ols(ols_url)
    generate_junit_report(suite_id, success)

    if not success:
        artifact_dir = os.getenv("ARTIFACT_DIR", ".")
        # Set the environment variables for must_gather
        os.environ["ARTIFACT_DIR"] = artifact_dir
        os.environ["SUITE_ID"] = suite_id
        must_gather()

    exit(0 if success else 1)
