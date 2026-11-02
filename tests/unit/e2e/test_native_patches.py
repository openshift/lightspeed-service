"""Test native patch payloads without JSON serialization or cluster access."""

from unittest.mock import Mock

import pytest
import yaml

from tests.e2e.utils import adapt_ols_config, cluster_api, ols_installer
from tests.e2e.utils import data_collector_control as collector


@pytest.mark.parametrize(
    "image", ["registry.test/service:latest", 'image"with\\quotes']
)
@pytest.mark.parametrize("installer", [True, False])
def test_image_patch_preserves_value(
    monkeypatch: pytest.MonkeyPatch, image: str, installer: bool
) -> None:
    """Pass image values unchanged in a native JSON patch list."""
    patch = Mock()
    monkeypatch.setattr(cluster_api, "patch_resource", patch)
    monkeypatch.setattr(cluster_api, "scale_deployment", Mock())
    module = ols_installer if installer else adapt_ols_config
    monkeypatch.setattr(
        module, "retry_until_timeout_or_success", Mock(return_value=True)
    )
    if installer:
        ols_installer.replace_ols_image(image)
    else:
        monkeypatch.setattr(
            module.cluster_utils, "deployment_exists", Mock(return_value=True)
        )
        monkeypatch.setattr(module, "disconnected", True)
        monkeypatch.setattr(module, "wait_for_deployment", Mock())
        adapt_ols_config._apply_e2e_specific_config(image)
    patch.assert_called_once_with(
        "deployment",
        "lightspeed-app-server",
        [
            {
                "op": "replace",
                "path": "/spec/template/spec/containers/0/image",
                "value": image,
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
