"""Utilities for controlling the data exporter in e2e tests.

This module provides methods to configure the lightspeed-to-dataverse-exporter
for testing purposes.
"""

import json
import os
import time

import yaml

from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils.constants import OLS_USER_DATA_PATH
from tests.e2e.utils.wait_for_ols import wait_for_ols

# Exporter config map constants
EXPORTER_CONFIG_MAP_NAME = "lightspeed-exporter-config"
EXPORTER_CONFIG_FILENAME = "config.yaml"
EXPORTER_NAMESPACE = "openshift-lightspeed"


class DataCollectorControl:
    """Control mechanism for the data collector/exporter."""

    def __init__(self, pod_name: str | None = None):
        """Initialize the control mechanism.

        Args:
            pod_name: Name of the OLS pod. If None, will be auto-detected.
        """
        self.pod_name = pod_name or cluster_utils.get_pod_by_prefix()[0]
        self._original_exporter_config: dict | None = None
        self._expected_interval: int | None = None

    def clear_data_directory(self) -> None:
        """Clear the data directory to prevent collection.

        This removes all data that would be collected, effectively
        disabling collection for the current cycle.
        """
        try:
            cluster_utils.remove_dir(self.pod_name, f"{OLS_USER_DATA_PATH}/feedback")
            cluster_utils.remove_dir(self.pod_name, f"{OLS_USER_DATA_PATH}/transcripts")
            print("Data directories cleared")
        except Exception as e:
            print(f"Warning: Could not clear data directories: {e}")

    def update_exporter_config(  # noqa: C901  # pylint: disable=R0912
        self,
        collection_interval: int | None = None,
        ingress_server_url: str | None = None,
        data_dir: str | None = None,
        ingress_server_auth_token: str | None = None,
        log_level: str = "DEBUG",
    ) -> None:
        """Update exporter ConfigMap settings.

        Args:
            collection_interval: Collection interval in seconds.
            ingress_server_url: Ingress server URL (e.g., stage vs prod).
            data_dir: Data directory path.
            ingress_server_auth_token: Auth token for ingress server.
            log_level: Log level (default: "DEBUG" for test visibility).

        Note: This requires container restart to take effect.
        """
        try:
            # Get the current config map (created by operator)
            result = cluster_utils.run_oc(
                [
                    "get",
                    "configmap",
                    EXPORTER_CONFIG_MAP_NAME,
                    "-n",
                    EXPORTER_NAMESPACE,
                    "-o",
                    "yaml",
                ]
            )
            configmap = yaml.safe_load(result.stdout)

            # Store original config for restoration (only first time)
            if self._original_exporter_config is None:
                self._original_exporter_config = configmap.copy()

            # Load the exporter config as YAML, update values, and save back
            exporter_config = yaml.safe_load(
                configmap["data"][EXPORTER_CONFIG_FILENAME]
            )

            # Update only the provided values
            if collection_interval is not None:
                exporter_config["collection_interval"] = collection_interval
            if ingress_server_url is not None:
                # Prevent accidental use of production ingress in CI tests
                if (
                    "console.redhat.com" in ingress_server_url
                    and "stage" not in ingress_server_url
                ):
                    raise ValueError(
                        f"Production ingress URL not allowed in tests: {ingress_server_url}"
                    )
                exporter_config["ingress_server_url"] = ingress_server_url
            if data_dir is not None:
                exporter_config["data_dir"] = data_dir
            if ingress_server_auth_token is not None:
                exporter_config["ingress_server_auth_token"] = ingress_server_auth_token
            if log_level is not None:
                exporter_config["log_level"] = log_level

            # Convert back to YAML string
            exporter_config_str = yaml.dump(exporter_config, default_flow_style=False)

            # Debug: show what we're about to patch
            print(f"ConfigMap YAML to be patched:\n{exporter_config_str}")

            # Patch the ConfigMap
            patch_data = json.dumps(
                {"data": {EXPORTER_CONFIG_FILENAME: exporter_config_str}}
            )
            result = cluster_utils.run_oc(
                [
                    "patch",
                    "configmap",
                    EXPORTER_CONFIG_MAP_NAME,
                    "-n",
                    EXPORTER_NAMESPACE,
                    "--type",
                    "merge",
                    "-p",
                    patch_data,
                ]
            )
            print(f"ConfigMap patch result: {result.stdout}")

            # Log what was updated
            print("Exporter config updated:")
            if collection_interval is not None:
                print(f"  - collection_interval: {collection_interval}s")
            if ingress_server_url is not None:
                print(f"  - ingress_server_url: {ingress_server_url}")
            if data_dir is not None:
                print(f"  - data_dir: {data_dir}")
            if ingress_server_auth_token is not None:
                print("  - ingress_server_auth_token: [SET]")
            if log_level is not None:
                print(f"  - log_level: {log_level}")
            print("Note: Container restart required for changes to take effect")
        except Exception as e:
            print(f"Warning: Could not modify exporter config: {e}")
            raise

    def set_exporter_collection_interval(self, interval_seconds: int) -> None:
        """Set the collection interval in the exporter config map.

        Args:
            interval_seconds: Collection interval in seconds.

        Note: This requires deployment restart to take effect.
              The operator does NOT reconcile the exporter ConfigMap if it exists.
        """
        # Store expected interval for verification
        self._expected_interval = interval_seconds

        # Scale down deployment first to ensure clean restart
        print("Scaling down deployment before ConfigMap update...")
        cluster_utils.run_oc(
            [
                "scale",
                "deployment/lightspeed-app-server",
                "-n",
                EXPORTER_NAMESPACE,
                "--replicas=0",
            ]
        )

        # Wait for pod to terminate completely
        print("Waiting for pod to terminate...")
        max_wait = 60
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                pods = cluster_utils.get_pod_by_prefix(fail_not_found=False)
                if not pods:
                    print("Pod terminated successfully")
                    break
            except Exception:
                print("Pod terminated (exception during check)")
                break
            time.sleep(2)

        # Extra wait to ensure pod is fully gone
        time.sleep(3)

        # Now update the ConfigMap
        # Note: The operator does NOT reconcile the exporter ConfigMap if it already exists
        print(f"Updating ConfigMap with collection_interval={interval_seconds}...")
        self.update_exporter_config(collection_interval=interval_seconds)

        # Wait for ConfigMap update to fully propagate in etcd
        print("Waiting for ConfigMap changes to propagate...")
        time.sleep(5)

    def restart_exporter_container(
        self, container_name: str = "lightspeed-to-dataverse-exporter"
    ) -> None:
        """Restart the exporter by scaling deployment back up.

        The deployment controller will create a new pod with the updated config.

        Args:
            container_name: Name of the exporter container (for verification).
        """
        try:
            print("Scaling deployment back up...")
            cluster_utils.run_oc(
                [
                    "scale",
                    "deployment/lightspeed-app-server",
                    "-n",
                    EXPORTER_NAMESPACE,
                    "--replicas=1",
                ]
            )
            print("Waiting for new pod to start...")
            time.sleep(5)

            # Wait for new pod to be ready
            max_wait = 60
            start_time = time.time()
            while time.time() - start_time < max_wait:
                try:
                    new_pod = cluster_utils.get_pod_by_prefix()[0]
                    # Verify pod is ready
                    result = cluster_utils.run_oc(
                        ["get", "pod", new_pod, "-n", EXPORTER_NAMESPACE, "-o", "json"]
                    )
                    pod_info = yaml.safe_load(result.stdout)
                    if pod_info.get("status", {}).get("phase") == "Running":
                        self.pod_name = new_pod
                        print(f"New pod {new_pod} is ready")

                        # Verify the exporter config was picked up
                        self._verify_config_applied(
                            new_pod, container_name, self._expected_interval
                        )

                        # Wait for OLS API to be ready (not just pod running)
                        print("Waiting for OLS API to be ready...")
                        ols_url = cluster_utils.get_ols_url("ols")
                        if not wait_for_ols(ols_url, timeout=120, interval=5):
                            print("Warning: OLS readiness check timed out")
                        else:
                            print("OLS API is ready")
                        return
                except Exception as e:
                    print(f"Waiting for pod... ({e})")
                time.sleep(2)

            print("Warning: Timeout waiting for new pod to be ready")
        except Exception as e:
            print(f"Warning: Could not restart container: {e}")
            raise

    def _verify_config_applied(
        self, pod_name: str, container_name: str, expected_interval: int | None = None
    ) -> None:
        """Verify the exporter picked up the new configuration.

        Args:
            pod_name: Name of the pod.
            container_name: Name of the exporter container.
            expected_interval: Expected collection interval in seconds.
        """
        try:
            # Get container logs to verify config
            log_result = cluster_utils.run_oc(
                [
                    "logs",
                    pod_name,
                    "-c",
                    container_name,
                    "-n",
                    EXPORTER_NAMESPACE,
                    "--tail=50",
                ]
            )
            logs = log_result.stdout

            # Look for collection interval in logs
            for line in logs.split("\n"):
                if "Collection interval:" in line:
                    print(f"Config verification: {line}")
                    if expected_interval is not None:
                        expected_str = f"{expected_interval} seconds"
                        if expected_str in line:
                            print(
                                f"✓ Collection interval matches expected: {expected_interval}s"
                            )
                        else:
                            print(
                                f"✗ WARNING: Expected {expected_interval}s but got: {line}"
                            )
                    return

            print("Warning: Could not verify collection interval in logs")
        except Exception as e:
            print(f"Warning: Could not verify config from logs: {e}")


def configure_exporter_for_e2e_tests(
    interval_seconds: int = 3600,
    ingress_env: str = "stage",
    cp_offline_token: str | None = None,
    log_level: str = "DEBUG",
    data_dir: str = "/app-root/ols-user-data",
) -> None:
    """Configure exporter for e2e tests with proper settings.

    Args:
        interval_seconds: Collection interval (default: 3600 = 1 hour).
        ingress_env: Ingress environment - "stage" or "prod" (default: "stage").
        cp_offline_token: Auth token for ingress server (required for stage).
        log_level: Log level (default: "DEBUG").
        data_dir: Data directory path (default: "/app-root/ols-user-data").
    """
    controller = DataCollectorControl()

    # Determine ingress URL based on environment
    if ingress_env == "stage":
        ingress_url = "https://console.stage.redhat.com/api/ingress/v1/upload"
    else:
        ingress_url = "https://console.redhat.com/api/ingress/v1/upload"

    # Get token from env if not provided
    if cp_offline_token is None:
        cp_offline_token = os.getenv("CP_OFFLINE_TOKEN", "")

    controller.update_exporter_config(
        collection_interval=interval_seconds,
        ingress_server_url=ingress_url,
        data_dir=data_dir,
        ingress_server_auth_token=cp_offline_token if cp_offline_token else None,
        log_level=log_level,
    )
    controller.restart_exporter_container()


def patch_exporter_mode_to_manual() -> None:
    """Patch the exporter container to use manual mode instead of openshift mode.

    In manual mode, the exporter uses the token from the ConfigMap instead of
    trying to get credentials from the OpenShift cluster. This is needed for
    CI testing where we use a stage ingress token.
    """
    # The exporter container args need to be changed from:
    #   ["--mode", "openshift", "--config", "/etc/config/config.yaml", ...]
    # to:
    #   ["--mode", "manual", "--config", "/etc/config/config.yaml", ...]
    patch = json.dumps(
        [
            {
                "op": "replace",
                "path": "/spec/template/spec/containers/1/args",
                "value": [
                    "--mode",
                    "manual",
                    "--config",
                    "/etc/config/config.yaml",
                    "--log-level",
                    "DEBUG",
                    "--data-dir",
                    "/app-root/ols-user-data",
                ],
            }
        ]
    )
    cluster_utils.run_oc(
        [
            "patch",
            "deployment/lightspeed-app-server",
            "-n",
            EXPORTER_NAMESPACE,
            "--type=json",
            "-p",
            patch,
        ]
    )
    print("Deployment patched: exporter now using manual mode")


def prepare_for_data_collection_test(
    short_interval_seconds: int = 5,
) -> DataCollectorControl:
    """Prepare the environment for testing data collection.

    This test runs in isolation with the 'data_export' marker, so we can:
    - Patch the deployment to use manual mode (uses ConfigMap token)
    - Set a short collection interval without worrying about other tests
    - No cleanup needed (operator will reconcile when it runs next)

    Args:
        short_interval_seconds: Collection interval for testing (default: 5s).

    Returns:
        DataCollectorControl instance for further operations.
    """
    controller = DataCollectorControl()

    # Get stage ingress URL and token
    stage_ingress_url = "https://console.stage.redhat.com/api/ingress/v1/upload"
    cp_offline_token = os.getenv("CP_OFFLINE_TOKEN", "")

    print(
        f"Setting up data export test with {short_interval_seconds}s collection interval"
    )
    print(f"Ingress URL: {stage_ingress_url}")
    if cp_offline_token:
        token_len = len(cp_offline_token)
        print(f"Auth token from CP_OFFLINE_TOKEN env var (len={token_len})")
    else:
        print("Warning: CP_OFFLINE_TOKEN not set, upload will fail with 401")

    controller._expected_interval = short_interval_seconds

    # Scale down deployment to apply changes
    print("Scaling down deployment...")
    cluster_utils.run_oc(
        [
            "scale",
            "deployment/lightspeed-app-server",
            "-n",
            EXPORTER_NAMESPACE,
            "--replicas=0",
        ]
    )

    # Wait for pod to terminate
    print("Waiting for pod to terminate...")
    max_wait = 60
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            pods = cluster_utils.get_pod_by_prefix(fail_not_found=False)
            if not pods:
                print("Pod terminated")
                break
        except Exception:
            break
        time.sleep(2)
    time.sleep(3)

    # Clear any existing data
    print("Clearing data directories...")
    controller.clear_data_directory()

    # Update ConfigMap
    print("Updating ConfigMap...")
    controller.update_exporter_config(
        collection_interval=short_interval_seconds,
        ingress_server_url=stage_ingress_url,
        ingress_server_auth_token=cp_offline_token if cp_offline_token else None,
    )

    # Patch deployment to use manual mode (uses ConfigMap token instead of cluster auth)
    print("Patching deployment to use manual mode...")
    patch_exporter_mode_to_manual()

    # Scale up and wait for pod
    controller.restart_exporter_container()

    # Wait for first collection cycle
    wait_time = short_interval_seconds + 3
    print(f"Waiting {wait_time}s for first collection cycle...")
    time.sleep(wait_time)

    return controller
