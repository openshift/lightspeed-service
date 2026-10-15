"""Unit tests for e2e OLS installation."""

import json
import subprocess
from threading import Barrier
from typing import Any

import pytest
import yaml

from tests.e2e.utils import cluster, ols_installer


@pytest.mark.parametrize(
    "provider, suffix, inspection_enabled, expected_url",
    [
        ("openai", "default", False, None),
        ("rhoai_vllm_lseval", "default", False, "https://model.example.test/v1"),
        ("openai", "mcp_inspection", True, None),
    ],
    ids=["default-template", "environment-substituted-template", "inspection-template"],
)
def test_install_ols_creates_olsconfig_with_suffix_inspection_value(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    suffix: str,
    inspection_enabled: bool,
    expected_url: str | None,
) -> None:
    """Pass the suffix-selected inspection CR to oc during a fresh install."""

    class OlsConfigCapturedError(Exception):
        """Stop installation after capturing the OLSConfig command."""

    captured: dict[str, Any] = {}

    def fake_run_oc(
        args: list[str],
        command: str | None = None,
        ignore_existing_resource: bool = False,
    ) -> Any:
        if args[:2] == ["create", "-f"]:
            captured["args"] = args
            captured["command"] = command
            captured["ignore_existing_resource"] = ignore_existing_resource
            raise OlsConfigCapturedError
        if args[:3] == ["get", "clusterserviceversion", "-o"]:
            return type("Result", (), {"stdout": "Succeeded"})()
        return None

    monkeypatch.setattr(ols_installer, "disconnected", True)
    monkeypatch.setattr(ols_installer.cluster_utils, "run_oc", fake_run_oc)
    monkeypatch.setattr(
        ols_installer, "create_and_config_sas", lambda: ("token", "metrics")
    )
    monkeypatch.setattr(ols_installer, "create_secrets", lambda *_args: None)
    monkeypatch.setattr(ols_installer, "warm_up_rhokp_image", lambda: None)
    monkeypatch.setattr(
        ols_installer, "retry_until_timeout_or_success", lambda *_args, **_kwargs: True
    )
    monkeypatch.setenv("PROVIDER", provider)
    monkeypatch.setenv("PROVIDER_KEY_PATH", "unused")
    monkeypatch.setenv("KSVC_URL", "https://model.example.test")
    monkeypatch.setenv("OLS_CONFIG_SUFFIX", suffix)

    with pytest.raises(OlsConfigCapturedError):
        ols_installer.install_ols()

    assert captured["args"] == ["create", "-f", "-"]
    assert captured["ignore_existing_resource"] is True
    cr = yaml.safe_load(captured["command"])
    assert (
        cr["spec"]["ols"]["guardrails"]["toolResultInspection"]["enabled"]
        is inspection_enabled
    )
    if expected_url is not None:
        provider_config = cr["spec"]["llm"]["providers"][0]
        assert provider_config["url"] == expected_url


def test_warm_up_rhokp_image_pulls_operator_image_concurrently_on_each_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pull the configured image on all nodes concurrently with a long timeout."""
    image = "registry.example.com/rhokp@sha256:abc123"
    deployment = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "manager",
                            "args": ["--leader-elect", f"--rhokp-image={image}"],
                        }
                    ]
                }
            }
        }
    }
    nodes = {
        "items": [
            {"metadata": {"name": "worker-a"}},
            {"metadata": {"name": "worker-b"}},
        ]
    }
    pull_barrier = Barrier(2)
    calls: list[tuple[list[str], int]] = []

    def fake_run_oc(
        args: list[str],
        command: str | None = None,
        ignore_existing_resource: bool = False,
        timeout: int = cluster.OC_SUBPROCESS_TIMEOUT,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((args, timeout))
        if args[:2] == ["get", "deployment"]:
            stdout = json.dumps(deployment)
        elif args[:2] == ["get", "nodes"]:
            stdout = json.dumps(nodes)
        else:
            pull_barrier.wait(timeout=2)
            stdout = ""
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(ols_installer.cluster_utils, "run_oc", fake_run_oc)

    ols_installer.warm_up_rhokp_image()

    pull_calls = [args for args, _ in calls if args[0] == "debug"]
    assert sorted(pull_calls) == sorted(
        [
            [
                "debug",
                "node/worker-a",
                "--",
                "chroot",
                "/host",
                "crictl",
                "pull",
                image,
            ],
            [
                "debug",
                "node/worker-b",
                "--",
                "chroot",
                "/host",
                "crictl",
                "pull",
                image,
            ],
        ]
    )
    assert [timeout for args, timeout in calls if args[0] == "debug"] == [600, 600]


def test_warm_up_rhokp_image_raises_when_deployment_has_no_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report a missing RHOKP image before attempting any node pulls."""
    calls: list[list[str]] = []

    def fake_run_oc(
        args: list[str],
        command: str | None = None,
        ignore_existing_resource: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(
            args,
            0,
            json.dumps(
                {"spec": {"template": {"spec": {"containers": [{"args": []}]}}}}
            ),
            "",
        )

    monkeypatch.setattr(ols_installer.cluster_utils, "run_oc", fake_run_oc)

    with pytest.raises(RuntimeError, match="--rhokp-image"):
        ols_installer.warm_up_rhokp_image()

    assert calls == [
        [
            "get",
            "deployment",
            "lightspeed-operator-controller-manager",
            "-n",
            "openshift-lightspeed",
            "-o",
            "json",
        ]
    ]


def test_warm_up_rhokp_image_raises_when_cluster_has_no_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report an empty node list instead of silently skipping the warm-up."""
    deployment = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "args": [
                                "--rhokp-image=registry.example.com/rhokp@sha256:abc123"
                            ]
                        }
                    ]
                }
            }
        }
    }
    outputs = iter([json.dumps(deployment), json.dumps({"items": []})])
    calls: list[list[str]] = []

    def fake_run_oc(
        args: list[str],
        command: str | None = None,
        ignore_existing_resource: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, next(outputs), "")

    monkeypatch.setattr(ols_installer.cluster_utils, "run_oc", fake_run_oc)

    with pytest.raises(RuntimeError, match="cluster nodes"):
        ols_installer.warm_up_rhokp_image()

    assert calls == [
        [
            "get",
            "deployment",
            "lightspeed-operator-controller-manager",
            "-n",
            "openshift-lightspeed",
            "-o",
            "json",
        ],
        ["get", "nodes", "-o", "json"],
    ]
