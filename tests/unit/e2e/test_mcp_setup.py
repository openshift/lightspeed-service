"""Unit tests for deploying the e2e MCP mock server."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.e2e.utils import mcp_setup


def test_deploy_mock_server_uses_current_repository_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply the checked-out server source in a ConfigMap before deployment."""
    commands: list[tuple[list[str], str | None]] = []

    def fake_run_oc(
        args: list[str], command: str | None = None, **_kwargs: Any
    ) -> None:
        commands.append((args, command))

    monkeypatch.setattr(mcp_setup.cluster_utils, "run_oc", fake_run_oc)
    monkeypatch.setattr(
        mcp_setup.cluster_utils,
        "get_pod_by_prefix",
        lambda **_kwargs: ["pod"],
    )
    monkeypatch.setattr(
        mcp_setup, "retry_until_timeout_or_success", lambda *_args, **_kwargs: True
    )

    mcp_setup._deploy_mock_server()

    configmap_command = next(
        command for args, command in commands if command is not None
    )
    configmap = yaml.safe_load(configmap_command)
    assert configmap["kind"] == "ConfigMap"
    assert configmap["data"]["server.py"] == Path(
        mcp_setup.SERVER_DIR / "server.py"
    ).read_text(encoding="utf-8")
    assert commands[0][0] == ["apply", "-f", "-"]
    assert commands[2][0] == ["apply", "-f", str(mcp_setup.DEPLOYMENT_YAML)]
    deployment = next(
        document
        for document in yaml.safe_load_all(
            mcp_setup.DEPLOYMENT_YAML.read_text(encoding="utf-8")
        )
        if document["kind"] == "Deployment"
    )
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == ["python", "-u", "/opt/mcp/server.py", "3000"]
