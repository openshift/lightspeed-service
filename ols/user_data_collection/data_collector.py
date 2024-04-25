"""Collect insights and upload it to the Ingress service.

It waits 5 min after startup before collects the data. Then it collects
data after specified interval.

When CP_OFFLINE_TOKEN is provided (either for prod or stage), it is used
for ingress authentication instead of cluster pull-secret.
"""

import base64
import io
import json
import logging
import os
import pathlib
import sys
import tarfile
import time
from typing import Any, Callable

import kubernetes
import requests

OLS_USER_DATA_PATH = os.environ["OLS_USER_DATA_PATH"]
OLS_USER_DATA_COLLECTION_INTERVAL = int(
    os.environ.get("OLS_USER_DATA_COLLECTION_INTERVAL", 2 * 60 * 60)
)  # 2 hours in seconds

INITIAL_WAIT = 60 * 5  # 5 minutes in seconds
RUN_WITHOUT_INITIAL_WAIT = bool(
    os.environ.get("RUN_WITHOUT_INITIAL_WAIT", "false").lower() == "true"
)

INGRESS_ENV = os.environ.get("INGRESS_ENV", "stage")  # prod/stage
if INGRESS_ENV not in {"prod", "stage"}:
    raise ValueError(
        f"Unknown value in INGRESS_ENV: {INGRESS_ENV}. Allowed: prod, stage"
    )
INGRESS_TIMEOUT = 30  # seconds
INGRESS_BASE_DELAY = 60  # exponential backoff parameter
INGRESS_MAX_RETRIES = 3  # exponential backoff parameter

CP_OFFLINE_TOKEN = os.environ.get("CP_OFFLINE_TOKEN")
REDHAT_SSO_TIMEOUT = 5  # seconds

# TODO: OLS-473
# OLS_USER_DATA_MAX_SIZE = 100 * 1024 * 1024  # 100 MB
USER_AGENT = "openshift-lightspeed-operator/user-data-collection cluster/{cluster_id}"

if INGRESS_ENV == "stage" and not CP_OFFLINE_TOKEN:
    raise ValueError("CP_OFFLINE_TOKEN is required for stage environment")


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "info").upper(),
    stream=sys.stdout,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
# silence libs
# - urllib3 - we don't care about those debug posts
# - kubernetes - prints resources content when debug, causing secrets leak
logging.getLogger("kubernetes").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class ClusterPullSecretNotFoundError(Exception):
    """Cluster pull-secret is not found."""


class ClusterIDNotFoundError(Exception):
    """Cluster id is not found."""


def get_ingress_upload_url() -> str:
    """Get the Ingress upload URL based on the environment."""
    upload_endpoint = "api/ingress/v1/upload"
    if INGRESS_ENV == "prod":
        return "https://console.redhat.com/" + upload_endpoint
    elif INGRESS_ENV == "stage":
        return "https://console.stage.redhat.com/" + upload_endpoint
    else:
        raise ValueError(f"Unknown value in INGRESS_ENV: {INGRESS_ENV}")


def access_token_from_offline_token(offline_token: str) -> str:
    """Generate "access token" from the "offline token".

    Offline token can be generated at:
        prod - https://access.redhat.com/management/api
        stage - https://access.stage.redhat.com/management/api

    Args:
        offline_token: Offline token from the Customer Portal.

    Returns:
        Refresh token.
    """
    if INGRESS_ENV == "stage":
        url = "https://sso.stage.redhat.com/"
    else:
        url = "https://sso.redhat.com/"
    endpoint = "auth/realms/redhat-external/protocol/openid-connect/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": "rhsm-api",
        "refresh_token": offline_token,
    }

    response = requests.post(url + endpoint, data=data, timeout=REDHAT_SSO_TIMEOUT)
    if response.status_code == requests.codes.ok:
        return response.json()["access_token"]
    else:
        raise Exception(f"Failed to generate access token. Response: {response.json()}")


def get_cloud_openshift_pull_secret() -> str:
    """Get the pull secret token from the cluster."""
    kubernetes.config.load_incluster_config()
    v1 = kubernetes.client.CoreV1Api()

    try:
        secret = v1.read_namespaced_secret("pull-secret", "openshift-config")
        dockerconfigjson = secret.data[".dockerconfigjson"]
        dockerconfig = json.loads(base64.b64decode(dockerconfigjson).decode("utf-8"))
        return dockerconfig["auths"]["cloud.openshift.com"]["auth"]
    except KeyError:
        logger.error("failed to get token from cluster pull-secret, missing keys")
    except TypeError:
        logger.error(
            "failed to get token from cluster pull-secret, unexpected "
            f"object type: {type(dockerconfig)}"
        )
    except kubernetes.client.exceptions.ApiException as e:
        logger.error(f"failed to get pull-secret object, body: {e.body}")
    raise ClusterPullSecretNotFoundError


def get_cluster_id() -> str:
    """Get the cluster_id from the cluster."""
    kubernetes.config.load_incluster_config()
    custom_objects_api = kubernetes.client.CustomObjectsApi()

    try:
        version_data = custom_objects_api.get_cluster_custom_object(
            "config.openshift.io", "v1", "clusterversions", "version"
        )
        return version_data["spec"]["clusterID"]
    except KeyError:
        logger.error(
            "failed to get cluster_id from cluster, missing keys in version object"
        )
    except TypeError:
        logger.error(f"failed to get cluster_id, version object is: {version_data}")
    except kubernetes.client.exceptions.ApiException as e:
        logger.error(f"failed to get version object, body: {e.body}")
    raise ClusterIDNotFoundError


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
    files += list(pathlib.Path(location).glob("transcripts/*/*/*.json"))

    return files


def package_files_into_tarball(
    file_paths: list[pathlib.Path], path_to_strip: str
) -> io.BytesIO:
    """Package specified directory into a tarball.

    Args:
        file_paths: List of paths to the files to be packaged.
        path_to_strip: Path to be stripped from the file paths (not
            included in the archive).

    Returns:
        BytesIO object representing the tarball.
    """
    tarball_io = io.BytesIO()

    with tarfile.open(fileobj=tarball_io, mode="w:gz") as tar:
        # arcname parameter is set to a stripped path to avoid including
        # the full path of the root dir
        for file_path in file_paths:
            # skip symlinks as those are a potential security risk
            if not file_path.is_symlink():
                tar.add(
                    file_path, arcname=file_path.as_posix().replace(path_to_strip, "")
                )

        # add magic file for identification of our archive on the CCX side
        empty_file = tarfile.TarInfo("openshift_lightspeed.json")
        empty_file.size = 0
        tar.addfile(empty_file)

    tarball_io.seek(0)

    return tarball_io


def exponential_backoff_decorator(max_retries: int, base_delay: int) -> Callable:
    """Exponential backoff decorator."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> None:
            retries = 0
            while retries < max_retries:
                try:
                    func(*args, **kwargs)
                    return
                except Exception as e:
                    logger.error(f"attempt {retries + 1} failed with error: {e}")
                    retries += 1
                    delay = base_delay * 2**retries
                    logger.info(f"retrying in {delay} seconds...")
                    time.sleep(delay)
            logger.error("max retries reached, operation failed.")

        return wrapper

    return decorator


@exponential_backoff_decorator(
    max_retries=INGRESS_MAX_RETRIES, base_delay=INGRESS_BASE_DELAY
)
def upload_data_to_ingress(tarball: io.BytesIO) -> requests.Response:
    """Upload the tarball to a Ingress.

    Args:
        tarball: BytesIO object representing the tarball to be uploaded.

    Returns:
        Response object from the Ingress.
    """
    logger.info("sending collected data")
    url = get_ingress_upload_url()
    payload = {
        "file": (
            "ols.tgz",
            tarball.read(),
            "application/vnd.redhat.openshift.periodic+tar",
        ),
    }

    if CP_OFFLINE_TOKEN:
        logger.debug("using CP offline token to generate refresh token")
        token = access_token_from_offline_token(CP_OFFLINE_TOKEN)
        # when authenticating with token, user-agent is not accepted
        # causing "UHC services authentication failed"
        headers = {"Authorization": f"Bearer {token}"}
    else:
        logger.debug("using cluster pull secret to authenticate")
        cluster_id = get_cluster_id()
        token = get_cloud_openshift_pull_secret()
        headers = {
            "User-Agent": USER_AGENT.format(cluster_id=cluster_id),
            "Authorization": f"Bearer {token}",
        }

    with requests.Session() as s:
        s.headers = headers
        logger.debug(f"posting payload to {url}")
        response = s.post(
            url=url,
            files=payload,
            timeout=INGRESS_TIMEOUT,
        )

    if response.status_code != 202:
        raise requests.exceptions.HTTPError(
            f"data upload failed with response: {response.json()}"
        )

    request_id = response.json()["request_id"]
    logger.info(f"data uploaded with request_id: '{request_id}'")

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
        logger.debug(f"collected files: {collected_files}")
        tarball = package_files_into_tarball(collected_files, path_to_strip=data_path)
        try:
            upload_data_to_ingress(tarball)
            delete_data(collected_files)
            logger.info("uploaded data removed")
        except (ClusterPullSecretNotFoundError, ClusterIDNotFoundError) as e:
            logger.error(f"{e.__class__.__name__} - upload and data removal canceled")
        # TODO: OLS-473
        # ensure_data_folder_is_not_bigger_than(OLS_USER_DATA_MAX_SIZE)
    else:
        logger.info(f"'{data_path}' contains no data, nothing to do...")


# NOTE: This condition is here mainly to have a way how to influence
# when the collector is running in the e2e tests. It is not meant to be
# used in the production.
def disabled_by_file() -> bool:
    """Check if the data collection is disabled by a file.

    Pure existence of the file `disable_collector` in the root of the
    user data dir is enough to disable the data collection.
    """
    return (pathlib.Path(OLS_USER_DATA_PATH) / "disable_collector").exists()


if __name__ == "__main__":
    if not RUN_WITHOUT_INITIAL_WAIT:
        logger.info(
            "collection script started, waiting 5 minutes before first collection"
        )
        time.sleep(INITIAL_WAIT)
    while True:
        if not disabled_by_file():
            gather_ols_user_data(OLS_USER_DATA_PATH)
        else:
            logger.info("disabled by control file, skipping data collection")
        time.sleep(OLS_USER_DATA_COLLECTION_INTERVAL)
