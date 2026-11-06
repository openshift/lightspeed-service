"""Collect cluster resources and pod logs in ARTIFACT_DIR/SUITE_ID."""

import os
import sys
from pathlib import Path

import yaml
from kubernetes.client import ApiException
from openshift.dynamic.exceptions import NotFoundError, ResourceNotFoundError

from ols.constants import DEFAULT_CONFIGURATION_FILE

sys.path.append(Path(__file__).parent.parent.parent.as_posix())
from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils import cluster_api
from tests.e2e.utils.cluster import PodNotFoundError


def must_gather() -> None:
    """Collect logs and state from the test namespace."""
    artifact_dir = os.environ.get("ARTIFACT_DIR")
    if not artifact_dir:
        print("$ARTIFACT_DIR is not set, skipping gathering")
        return
    suite_id = os.environ.get("SUITE_ID", "nosuite")
    cluster_dir = Path(artifact_dir) / suite_id / "cluster"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    for resource, filename in (
        ("pod", "pods.yaml"),
        ("service", "services.yaml"),
        ("deployment", "deployments.yaml"),
        ("replicaset", "replicasets.yaml"),
        ("route", "routes.yaml"),
        ("rolebinding", "rolebindings.yaml"),
        ("serviceaccount", "serviceaccounts.yaml"),
        ("olsconfig", DEFAULT_CONFIGURATION_FILE),
        ("clusterserviceversion", "clusterserviceversion.yaml"),
        ("installplan", "installplan.yaml"),
        ("configmap", "configmap.yaml"),
    ):
        try:
            resources = cluster_api.read_resource(resource)
        except (NotFoundError, ResourceNotFoundError):
            print(f"Resource type {resource} not found on cluster, skipping")
            continue
        (cluster_dir / filename).write_text(yaml.safe_dump(resources), encoding="utf-8")
    pod_logs_dir = cluster_dir / "podlogs"
    pod_logs_dir.mkdir(exist_ok=True)
    for pod in cluster_utils.get_running_pods():
        try:
            containers = cluster_utils.get_pod_containers(pod)
        except PodNotFoundError:
            print(f"Pod {pod} vanished before logs could be collected, skipping")
            continue
        for container in containers:
            try:
                logs = cluster_api.pod_logs(pod, container)
            except ApiException as error:
                if error.status != 404:
                    raise
                print(f"Pod {pod} vanished before logs could be collected, skipping")
                break
            (pod_logs_dir / f"{pod}-{container}.log").write_text(logs, encoding="utf-8")


if __name__ == "__main__":
    must_gather()
