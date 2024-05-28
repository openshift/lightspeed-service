# wait_for_ols.py

import time
import requests
import os
import warnings
import xml.etree.ElementTree as ET
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings('ignore', category=InsecureRequestWarning)

def wait_for_ols(url, timeout=600, interval=10):
    print("Starting wait_for_ols")
    attempts = int(timeout / interval)
    for attempt in range(1, attempts + 1):
        print(f"Checking OLS readiness, attempt {attempt} of {attempts}")
        try:
            response = requests.get(f"{url}/readiness", verify=False)
            if response.status_code == 200:
                print("OLS is ready")
                return True
        except requests.RequestException:
            pass
        time.sleep(interval)
    print("Timed out waiting for OLS to become available")
    return False

def generate_junit_report(suite_id, success):
    testsuite = ET.Element('testsuite', name=suite_id, tests="1", failures="0" if success else "1")
    testcase = ET.SubElement(testsuite, 'testcase', name="wait_for_ols", classname=f"{suite_id}.wait_for_ols")
    if not success:
        failure = ET.SubElement(testcase, 'failure', message="OLS failed to start up")
        failure.text = "OLS did not become available in time"
    tree = ET.ElementTree(testsuite)
    artifact_dir = os.getenv("ARTIFACT_DIR", ".")
    tree.write(f'{artifact_dir}/junit_setup_{suite_id}.xml')

if __name__ == "__main__":
    suite_id = os.getenv('SUITE_ID')
    ols_url = os.getenv('OLS_URL')
    if not ols_url:
        print("OLS_URL environment variable is not set")
        generate_junit_report(suite_id, False)
        exit(1)
    success = wait_for_ols(ols_url)
    generate_junit_report(suite_id, success)
    if not success:
        artifact_dir = os.getenv("ARTIFACT_DIR", ".")
        os.system(f"python tests/scripts/must_gather.py ARTIFACT_DIR={artifact_dir} SUITE_ID={suite_id}")
    exit(0 if success else 1)