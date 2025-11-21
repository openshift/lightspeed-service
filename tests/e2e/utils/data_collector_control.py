"""Utilities for controlling the data exporter in e2e tests.

This module provides methods to configure the lightspeed-to-dataverse-exporter
for testing purposes.
"""

import time

import yaml

from tests.e2e.utils import cluster as cluster_utils

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

    def clear_data_directory(self) -> None:
        """Clear the data directory to prevent collection.

        This removes all data that would be collected, effectively
        disabling collection for the current cycle.
        """
        from tests.e2e.utils.constants import OLS_USER_DATA_PATH

        try:
            cluster_utils.remove_dir(self.pod_name, f"{OLS_USER_DATA_PATH}/feedback")
            cluster_utils.remove_dir(self.pod_name, f"{OLS_USER_DATA_PATH}/transcripts")
            print("Data directories cleared")
        except Exception as e:
            print(f"Warning: Could not clear data directories: {e}")

    def update_exporter_config(
        self,
        collection_interval: int | None = None,
        ingress_server_url: str | None = None,
        data_dir: str | None = None,
        ingress_server_auth_token: str | None = None,
        log_level: str | None = None,
    ) -> None:
        """Update exporter ConfigMap settings.

        Args:
            collection_interval: Collection interval in seconds.
            ingress_server_url: Ingress server URL (e.g., stage vs prod).
            data_dir: Data directory path.
            ingress_server_auth_token: Auth token for ingress server.
            log_level: Log level (e.g., "debug", "info").

        Note: This requires container restart to take effect.
        """
        try:
            # Get the current config map
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

            # Parse the exporter config
            exporter_config_str = configmap["data"][EXPORTER_CONFIG_FILENAME]
            exporter_config = yaml.safe_load(exporter_config_str)

            # Update fields if provided
            updates = {
                "collection_interval": collection_interval,
                "ingress_server_url": ingress_server_url,
                "data_dir": data_dir,
                "ingress_server_auth_token": ingress_server_auth_token,
                "log_level": log_level,
            }
            for key, value in updates.items():
                if value is not None:
                    exporter_config[key] = value

            # Update the config map
            configmap["data"][EXPORTER_CONFIG_FILENAME] = yaml.dump(exporter_config)
            updated_yaml = yaml.dump(configmap)
            cluster_utils.run_oc(["apply", "-f", "-"], command=updated_yaml)

            # Log what was updated
            print("Exporter config updated:")
            for key, value in updates.items():
                if value is not None:
                    if key == "ingress_server_auth_token":
                        print("  - ingress_server_auth_token: [SET]")
                    else:
                        suffix = "s" if key == "collection_interval" else ""
                        print(f"  - {key}: {value}{suffix}")
            print("Note: Container restart required for changes to take effect")
        except Exception as e:
            print(f"Warning: Could not modify exporter config: {e}")
            raise

    def set_exporter_collection_interval(self, interval_seconds: int) -> None:
        """Set the collection interval in the exporter config map.

        Args:
            interval_seconds: Collection interval in seconds.

        Note: This requires container restart to take effect.
        """
        self.update_exporter_config(collection_interval=interval_seconds)

    def restart_exporter_container(
        self, container_name: str = "lightspeed-to-dataverse-exporter"
    ) -> None:
        """Restart the exporter container by deleting the pod.

        The deployment controller will recreate the pod with the new config.

        Args:
            container_name: Name of the exporter container (for verification).
        """
        try:
            print(f"Restarting pod {self.pod_name} to apply config changes...")
            cluster_utils.run_oc(
                ["delete", "pod", self.pod_name, "-n", EXPORTER_NAMESPACE]
            )
            print("Pod deleted, waiting for new pod to start...")
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
                        return
                except Exception as e:
                    print(f"Waiting for pod... ({e})")
                time.sleep(2)

            print("Warning: Timeout waiting for new pod to be ready")
        except Exception as e:
            print(f"Warning: Could not restart container: {e}")
            raise


def configure_exporter_for_e2e_tests(
    interval_seconds: int = 3600,
    ingress_env: str = "stage",
    cp_offline_token: str | None = None,
    log_level: str = "debug",
    data_dir: str = "/app-root/ols-user-data",
) -> None:
    """Configure exporter for e2e tests with proper settings.

    Args:
        interval_seconds: Collection interval (default: 3600 = 1 hour).
        ingress_env: Ingress environment - "stage" or "prod" (default: "stage").
        cp_offline_token: Auth token for ingress server (required for stage).
        log_level: Log level (default: "debug").
        data_dir: Data directory path (default: "/app-root/ols-user-data").
    """
    import os

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


def prepare_for_data_collection_test(
    short_interval_seconds: int = 10,
) -> DataCollectorControl:
    """Prepare the environment for testing data collection.

    This function:
    1. Clears existing data directories
    2. Sets a short collection interval for fast testing
    3. Restarts the exporter container to apply changes

    Args:
        short_interval_seconds: Short collection interval for testing (default: 10s).

    Returns:
        DataCollectorControl instance for further operations.
    """
    controller = DataCollectorControl()

    # Clear any existing data
    print("Clearing data directories before test...")
    controller.clear_data_directory()

    # Set short collection interval for testing
    print(f"Setting collection interval to {short_interval_seconds}s for testing...")
    controller.set_exporter_collection_interval(short_interval_seconds)

    # Restart exporter to apply new config
    controller.restart_exporter_container()

    # Wait a bit for the container to fully start
    print("Waiting for exporter to start with new config...")
    time.sleep(5)

    return controller


def cleanup_after_data_collection_test(
    controller: DataCollectorControl, restore_interval_seconds: int = 3600
) -> None:
    """Clean up after data collection test.

    Restores a long collection interval to prevent interference with other tests.

    Args:
        controller: DataCollectorControl instance from prepare_for_data_collection_test.
        restore_interval_seconds: Interval to restore (default: 3600s = 1 hour).
    """
    print(f"Restoring collection interval to {restore_interval_seconds}s...")
    controller.set_exporter_collection_interval(restore_interval_seconds)
    controller.restart_exporter_container()
    print("Data collection test cleanup complete")
