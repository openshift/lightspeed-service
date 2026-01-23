"""This module contains the wait_for_ols function and generate_junit_report function.

to check the readiness of the OLS and generate a JUnit report.
"""

import time
import warnings

import requests
from requests.exceptions import SSLError
from urllib3.exceptions import InsecureRequestWarning
from tests.e2e.utils.constants import (
    BASIC_ENDPOINTS_TIMEOUT,
)

warnings.filterwarnings("ignore", category=InsecureRequestWarning)


# ruff: noqa: S501
def wait_for_ols(url, client, timeout=300, interval=10):
    """Wait for the OLS to become ready by checking its readiness endpoint.

    Args:
        url (str): The base URL of the OLS service.
        timeout (int, optional): The maximum time to wait in seconds. Default is 600.
        interval (int, optional): The interval between readiness checks in seconds. Default is 10.

    Returns:
        bool: True if OLS becomes ready within the timeout, False otherwise.
    """
    print(f"Starting wait_for_ols at url {url}")
    attempts = int(timeout / interval)
    for attempt in range(1, attempts + 1):
        print(f"Checking OLS readiness, attempt {attempt} of {attempts}")
        try:
            response = client.get(
                    "/readiness",
                    timeout=BASIC_ENDPOINTS_TIMEOUT
                    )
            if response.status_code == requests.codes.ok:
                print("OLS is ready")
                return True
        except SSLError:
            print("SSL error detected, retrying without SSL verification")
            try:
                response = client.get(
                    "/readiness",
                    timeout=BASIC_ENDPOINTS_TIMEOUT
                    )
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
