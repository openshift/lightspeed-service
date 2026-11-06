"""Test deployment of the checked-out MCP mock server."""

from unittest.mock import Mock, call

import pytest
import yaml

from tests.e2e.utils import cluster_api, mcp_setup


def test_deploy_mock_server_uses_current_repository_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply source before replacing the server deployment."""
    calls = Mock()
    for operation in ("apply_resource", "delete_resource", "apply_file"):
        monkeypatch.setattr(cluster_api, operation, getattr(calls, operation))
    monkeypatch.setattr(
        mcp_setup, "retry_until_timeout_or_success", lambda *_a, **_k: True
    )
    mcp_setup._deploy_mock_server()
    body = calls.apply_resource.call_args.args[0]
    assert body["kind"] == "ConfigMap"
    assert body["data"]["server.py"] == (mcp_setup.SERVER_DIR / "server.py").read_text(
        encoding="utf-8"
    )
    assert [call[0] for call in calls.mock_calls] == [
        "delete_resource",
        "apply_resource",
        "delete_resource",
        "apply_file",
    ]
    assert calls.delete_resource.call_args_list[0].args[:2] == (
        "configmap",
        mcp_setup.MOCK_SERVER_SOURCE_CONFIGMAP,
    )
    assert calls.delete_resource.call_args_list[1] == call(
        "deployment",
        mcp_setup.MOCK_SERVER_NAME,
        mcp_setup.NAMESPACE,
        ignore_missing=True,
    )
    calls.apply_file.assert_called_once_with(str(mcp_setup.DEPLOYMENT_YAML))
    deployment = next(
        doc
        for doc in yaml.safe_load_all(mcp_setup.DEPLOYMENT_YAML.read_text())
        if doc["kind"] == "Deployment"
    )
    assert deployment["spec"]["template"]["spec"]["containers"][0]["command"] == [
        "python",
        "-u",
        "/opt/mcp/server.py",
        "3000",
    ]
