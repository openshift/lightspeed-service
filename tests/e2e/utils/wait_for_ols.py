"""This module contains the wait_for_ols function and generate_junit_report function.

to check the readiness of the OLS and generate a JUnit report.
"""

import time
import warnings

import requests
from requests.exceptions import SSLError
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)


# ruff: noqa: S501
def wait_for_ols(url, timeout=300, interval=10):
    """Wait for the OLS to become ready by checking its readiness endpoint.

    Args:
        url (str): The base URL of the OLS service.
        timeout (int, optional): The maximum time to wait in seconds. Default is 300.
        interval (int, optional): The interval between readiness checks in seconds. Default is 10.

    Returns:
        bool: True if OLS becomes ready within the timeout, False otherwise.
    """
    print(f"Starting wait_for_ols at url {url}")
    request_timeout = 35
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        remaining = deadline - time.monotonic()
        print(
            f"Checking OLS readiness, attempt {attempt} ({int(remaining)}s remaining)"
        )
        try:
            response = requests.get(
                f"{url}/readiness", verify=True, timeout=min(request_timeout, remaining)
            )
            if response.status_code == requests.codes.ok:
                print("OLS is ready")
                return True
            print(f"OLS not ready, status code: {response.status_code}")
        except SSLError as e:
            print(f"SSL error detected: {e}")
            print("Retrying without SSL verification")
            try:
                response = requests.get(
                    f"{url}/readiness",
                    verify=False,
                    timeout=min(request_timeout, remaining),
                )
                if response.status_code == requests.codes.ok:
                    print("OLS is ready")
                    return True
                print(f"OLS not ready (no-verify), status code: {response.status_code}")
            except requests.RequestException as retry_err:
                print(f"Request failed after SSL retry: {retry_err}")
        except requests.RequestException as e:
            print(f"Request failed: {e}")
        if time.monotonic() >= deadline:
            break
        time.sleep(min(interval, deadline - time.monotonic()))
    print("Timed out waiting for OLS to become available")
    return False
