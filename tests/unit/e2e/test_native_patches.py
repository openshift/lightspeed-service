"""Test native patch payloads without JSON serialization or cluster access."""

from unittest.mock import Mock

import pytest
import yaml

from tests.e2e.utils import cluster_api, ols_installer
from tests.e2e.utils import data_collector_control as collector


@pytest.mark.parametrize(
    "image", ["registry.test/service:latest", 'image"with\\quotes']
)
def test_image_patch_preserves_value(
    monkeypatch: pytest.MonkeyPatch, image: str
) -> None:
    """Pass candidate images to the operator without altering their values."""
    monkeypatch.setenv("OLS_IMAGE", image)
    deployment = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "manager",
                            "args": ["--service-image=old", "--leader-elect"],
                        }
                    ]
                }
            }
        }
    }
    monkeypatch.setattr(cluster_api, "read_resource", Mock(return_value=deployment))
    patch = Mock()
    monkeypatch.setattr(cluster_api, "patch_resource", patch)
    monkeypatch.setattr(
        ols_installer, "retry_until_timeout_or_success", Mock(return_value=True)
    )
    ols_installer.configure_operator_service_image()
    patch.assert_called_once_with(
        "deployment",
        "lightspeed-operator-controller-manager",
        [
            {
                "op": "add",
                "path": "/spec/template/spec/containers/0/args",
                "value": ["--leader-elect", f"--service-image={image}"],
            }
        ],
        patch_type="json",
    )


def test_exporter_config_patch_preserves_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch only ConfigMap data and preserve quoted configuration values."""
    monkeypatch.setattr(
        cluster_api,
        "read_resource",
        Mock(return_value={"data": {collector.EXPORTER_CONFIG_FILENAME: "{}"}}),
    )
    patch = Mock()
    monkeypatch.setattr(cluster_api, "patch_resource", patch)
    data_dir = '/data/"quoted"\\directory'
    collector.DataCollectorControl("pod").update_exporter_config(data_dir=data_dir)
    args = patch.call_args.args
    assert args[:2] == ("configmap", collector.EXPORTER_CONFIG_MAP_NAME)
    assert args[3] == collector.EXPORTER_NAMESPACE
    assert set(args[2]) == {"data"}
    assert yaml.safe_load(args[2]["data"][collector.EXPORTER_CONFIG_FILENAME]) == {
        "data_dir": data_dir,
        "log_level": "DEBUG",
    }
    assert patch.call_count == 1


def test_exporter_manual_mode_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass exporter arguments as a native JSON patch list."""
    patch = Mock()
    monkeypatch.setattr(cluster_api, "patch_resource", patch)
    collector.patch_exporter_mode_to_manual()
    patch.assert_called_once_with(
        "deployment",
        "lightspeed-app-server",
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
        ],
        collector.EXPORTER_NAMESPACE,
        patch_type="json",
    )
