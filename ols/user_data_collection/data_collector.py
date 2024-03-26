"""Collect insights and upload it to the Ingress service.

It collects the data on startup and then it waits for the specified interval.
"""

import io
import logging
import os
import pathlib
import tarfile
import time

import requests

OLS_USER_DATA_COLLECTION_INTERVAL = int(
    os.environ.get("UDC_COLLECTION_INTERVAL", 2 * 60 * 60)
)  # 2 hours in seconds
OLS_USER_DATA_PATH = os.environ.get(
    "UDC_STORAGE", "/home/ometelka/projects/lightspeed-service/user-data/"
)  # TODO: remove my default

# available ingress urls are:
# - https://console.redhat.com/ - production
# - https://console.stage.redhat.com/ - staging/dev
# NOTE: Token is not working for stage...
INGRESS_URL = os.environ.get("INGRESS_URL", "https://console.redhat.com/")
INGRESS_ENDPOINT = os.environ.get("INGRESS_ENDPOINT", "api/ingress/v1/upload")
INGRESS_TIMEOUT = 10  # seconds

# customer portal offline token is generated from https://access.redhat.com/management/api
CP_OFFLINE_TOKEN = os.environ.get(
    "CP_OFFLINE_TOKEN",
    "...",
)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def refresh_token_from_offline_token(offline_token: str) -> str:
    """Generate "refresh token" from the "offline token".

    Offline token can be generated at:
        https://access.redhat.com/management/api

    Args:
        offline_token: Offline token from the Customer Portal.

    Returns:
        Refresh token.
    """
    url = "https://sso.redhat.com/auth/realms/redhat-external/protocol/openid-connect/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": "rhsm-api",
        "refresh_token": offline_token,
    }

    response = requests.post(url, data=data, timeout=INGRESS_TIMEOUT)
    if response.status_code == requests.codes.ok:
        return response.json()["access_token"]
    else:
        # TODO: better error handling
        raise Exception("Failed to generate refresh token.")


def collect_ols_data_from(location: str) -> list[pathlib.Path]:
    """Collect files from a given location.

    Args:
        location: Path to the directory to be searched for files.

    Returns:
        List of paths to the collected files.

    Only JSON files from the 'feedback' and 'transcripts' directories are collected.
    """
    files = []

    files += list(pathlib.Path(location).glob("feedback/*.json"))
    files += list(pathlib.Path(location).glob("transcripts/*.json"))

    return files


def package_files_into_tarball(
    file_paths: list[pathlib.Path], path_to_strip: str = OLS_USER_DATA_PATH
) -> io.BytesIO:
    """Package specified directory into a tarball.

    Args:
        file_paths: List of paths to the files to be packaged.
        path_to_strip: Path to be stripped from the file paths (not
            include in the archive).

    Returns:
        BytesIO object representing the tarball.
    """
    tarball_io = io.BytesIO()

    with tarfile.open(fileobj=tarball_io, mode="w:gz") as tar:
        # arcname parameter is set to a stripped path to avoid including
        # the full path of the root dir
        for file_path in file_paths:
            tar.add(file_path, arcname=file_path.as_posix().replace(path_to_strip, ""))

        # add magic file for identification of our archive on the CCX side
        with open("openshift_lightspeed.json", "w") as magic_file:
            magic_file.write("{}")
        tar.add("openshift_lightspeed.json")

    tarball_io.seek(0)
    return tarball_io


def upload_data_to_ingress(tarball: io.BytesIO) -> requests.Response:
    """Upload the tarball to a Ingress.

    Args:
        tarball: BytesIO object representing the tarball to be uploaded.

    Returns:
        Response object from the Ingress.
    """
    logger.info("sending collected data")
    payload = {
        "file": (
            "ols.tgz",
            tarball.read(),
            "application/vnd.redhat.openshift.periodic+tar",
        ),
    }

    token = refresh_token_from_offline_token(CP_OFFLINE_TOKEN)

    response = requests.post(
        INGRESS_URL + INGRESS_ENDPOINT,
        files=payload,
        timeout=INGRESS_TIMEOUT,
        headers={
            # TODO: figure out the right user-agent
            "User-Agent": "ols-test",
            "Authorization": f"Bearer {token}",
        },
    )
    return response


def delete_data(file_paths: list[pathlib.Path]) -> None:
    """Delete files from the provided paths.

    Args:
        file_paths: List of paths to the files to be deleted.
    """
    for file_path in file_paths:
        logger.debug(f"removing '{file_path}'")
        file_path.unlink()
        if file_path.exists():
            logger.error(f"failed to remove '{file_path}'")


def gather_ols_user_data(data_path: str) -> None:
    """Gather OLS user data and upload it to the Ingress service."""
    collected_files = collect_ols_data_from(data_path)
    if collected_files:
        logger.info(f"collected {len(collected_files)} files from '{data_path}'")
        tarball = package_files_into_tarball(collected_files)
        response = upload_data_to_ingress(tarball)
        if response.status_code == 202:
            logger.info(
                f"data uploaded with request_id: '{response.json()['request_id']}'"
            )
            delete_data(collected_files)
            logger.info("uploaded data removed")
        else:
            # TODO: exponential backoff
            logger.info("data upload failed, schedulling retries...")
        # TODO: is this needed? Maybe check the total size instead?
        # delete_data_older_than_24_hours()
    else:
        logger.info(f"'{data_path}' contains no data, nothing to do...")


if __name__ == "__main__":
    while True:
        gather_ols_user_data(OLS_USER_DATA_PATH)
        time.sleep(OLS_USER_DATA_COLLECTION_INTERVAL)
