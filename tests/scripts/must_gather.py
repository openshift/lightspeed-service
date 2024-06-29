"""A utility for CI e2e tests, collecting artifacts such as logs and states.

Configuration is via the `ARTIFACT_DIR` and `SUITE_ID` environment variables.
"""

import os
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.parent.as_posix())
from tests.e2e.utils import cluster as cluster_utils


def must_gather():
    """Collect logs and state from openshift-lightspeed namespace."""
    artifact_dir = os.environ.get("ARTIFACT_DIR")
    if not artifact_dir:
        print("$ARTIFACT_DIR is not set, skipping gathering")
        return
    suite_id = os.environ.get("SUITE_ID", "nosuite")
    cluster_dir = Path(artifact_dir) / suite_id / "cluster"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    # pods, services, deployments, replica sets, routes.
    cluster_utils.run_oc_and_store_stdout(
        [
            "get",
            "all",
            "-n",
            "openshift-lightspeed",
            "-o",
            "yaml",
        ],
        f"{cluster_dir.as_posix()}/resources.yaml",
    )

    # olsconfig CR
    cluster_utils.run_oc_and_store_stdout(
        [
            "get",
            "olsconfig",
            "-n",
            "openshift-lightspeed",
            "-o",
            "yaml",
        ],
        f"{cluster_dir.as_posix()}/olsconfig.yaml",
    )

    # clusterserviceversion
    cluster_utils.run_oc_and_store_stdout(
        [
            "get",
            "clusterserviceversion",
            "-n",
            "openshift-lightspeed",
            "-o",
            "yaml",
        ],
        f"{cluster_dir.as_posix()}/clusterserviceversion.yaml",
    )

    # installplan
    cluster_utils.run_oc_and_store_stdout(
        [
            "get",
            "installplan",
            "-n",
            "openshift-lightspeed",
            "-o",
            "yaml",
        ],
        f"{cluster_dir.as_posix()}/installplan.yaml",
    )

    # pod logs
    pod_logs_dir = cluster_dir / "podlogs"
    pod_logs_dir.mkdir(exist_ok=True)
    for pod in cluster_utils.get_pods():
        for container in cluster_utils.get_pod_containers(pod):
            cluster_utils.run_oc_and_store_stdout(
                [
                    "logs",
                    f"pod/{pod}",
                    "-c",
                    container,
                ],
                f"{pod_logs_dir.as_posix()}/{pod}-{container}.log",
            )


if __name__ == "__main__":
    print("Running must_gather...")
    must_gather()
