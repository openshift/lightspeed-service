"""Unit tests for e2e OLS configuration adaptation."""

import json
import subprocess
from io import StringIO
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

from ols.constants import DEFAULT_CONFIGURATION_FILE
from tests.e2e import conftest as e2e_conftest
from tests.e2e.utils import adapt_ols_config, olsconfig_cr


@pytest.mark.parametrize(
    "provider_list",
    [["openai"], ["openai", "watsonx"]],
    ids=["single-provider", "multiple-providers"],
)
def test_apply_olsconfig_sets_inspection_from_suffix_in_both_orders(
    monkeypatch: pytest.MonkeyPatch, provider_list: list[str]
) -> None:
    """Set inspection only for the dedicated suffix without leaking between runs."""
    cr = {
        "apiVersion": "ols.openshift.io/v1alpha1",
        "kind": "OLSConfig",
        "metadata": {"name": "cluster"},
        "spec": {"ols": {"defaultProvider": "openai"}},
    }
    applied: list[dict[str, Any]] = []

    def fake_run_oc(
        args: list[str],
        command: str | None = None,
        ignore_existing_resource: bool = False,
    ) -> None:
        assert command is not None
        applied.append(
            {
                "args": args,
                "config": yaml.safe_load(command),
                "ignore_existing_resource": ignore_existing_resource,
            }
        )

    monkeypatch.setattr(
        olsconfig_cr,
        "open",
        lambda *_args, **_kwargs: StringIO(yaml.safe_dump(cr)),
        raising=False,
    )
    monkeypatch.setattr(adapt_ols_config.cluster_utils, "run_oc", fake_run_oc)

    for suffix, expected_enabled in (
        ("mcp_inspection", True),
        ("mcp_inspection_gpt_6_luna", True),
        ("default", False),
        ("default", False),
        ("mcp_inspection", True),
    ):
        monkeypatch.setenv("OLS_CONFIG_SUFFIX", suffix)
        adapt_ols_config.apply_olsconfig(provider_list)
        assert (
            applied[-1]["config"]["spec"]["ols"]["guardrails"]["toolResultInspection"][
                "enabled"
            ]
            is expected_enabled
        )

    assert all(item["args"] == ["apply", "-f", "-"] for item in applied)
    assert all(
        item["ignore_existing_resource"] is (len(provider_list) > 1) for item in applied
    )


@pytest.mark.parametrize(
    ("suffix", "expected_names"),
    [
        ("mcp_inspection", ["inspection", "ordinary"]),
        ("mcp_inspection_gpt_6_luna", ["inspection", "ordinary"]),
        ("default", ["ordinary"]),
    ],
)
def test_collection_keeps_inspection_tests_only_for_inspection_suffix(
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    expected_names: list[str],
) -> None:
    """Select inspection-marked tests for base and model-specific suffixes."""
    monkeypatch.setenv("OLS_CONFIG_SUFFIX", suffix)
    config = Mock()
    inspection_item = Mock()
    inspection_item.name = "inspection"
    inspection_item.config = config
    inspection_item.get_closest_marker.side_effect = lambda marker: (
        object() if marker == "inspection" else None
    )
    ordinary_item = Mock()
    ordinary_item.name = "ordinary"
    ordinary_item.config = config
    ordinary_item.get_closest_marker.return_value = None
    items = [inspection_item, ordinary_item]

    e2e_conftest.pytest_collection_modifyitems(items)

    assert [item.name for item in items] == expected_names


def test_apply_olsconfig_uses_inspection_template_when_suffix_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load the dedicated inspection CR for the inspection suffix."""
    applied: dict[str, Any] = {}

    def fake_run_oc(
        args: list[str],
        command: str | None = None,
        ignore_existing_resource: bool = False,
    ) -> None:
        applied["args"] = args
        applied["config"] = yaml.safe_load(command)

    monkeypatch.setenv("OLS_CONFIG_SUFFIX", "mcp_inspection")
    monkeypatch.setattr(adapt_ols_config.cluster_utils, "run_oc", fake_run_oc)

    adapt_ols_config.apply_olsconfig(["openai"])

    ols_spec = applied["config"]["spec"]["ols"]
    assert applied["args"] == ["apply", "-f", "-"]
    assert ols_spec["guardrails"]["toolResultInspection"]["enabled"] is True
    assert ols_spec["toolsApprovalConfig"]["approvalType"] == "never"


def _configmap_run_oc_stub(
    config: dict[str, Any], calls: list[tuple[list[str], str | None]]
) -> Any:
    """Build a run_oc stub serving the olsconfig configmap and pod list."""

    def fake_run_oc(
        args: list[str],
        command: str | None = None,
        ignore_existing_resource: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((args, command))
        if args[0] == "get" and args[1].startswith("cm"):
            stdout = yaml.safe_dump(config)
        elif args[:2] == ["get", "pods"]:
            stdout = json.dumps({"items": []})
        else:
            stdout = ""
        return subprocess.CompletedProcess(args, 0, stdout, "")

    return fake_run_oc


def _olsconfig_configmap(default_provider: str, default_model: str) -> dict[str, Any]:
    """Build an olsconfig configmap with the given defaults."""
    service_config = {
        "llm_config": {
            "default_provider": default_provider,
            "default_model": default_model,
        },
        "ols_config": {},
    }
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "olsconfig", "namespace": "openshift-lightspeed"},
        "data": {DEFAULT_CONFIGURATION_FILE: yaml.safe_dump(service_config)},
    }


def test_switch_default_provider_is_noop_when_already_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid a restart when the requested pair is already the default."""
    calls: list[tuple[list[str], str | None]] = []
    monkeypatch.setattr(
        adapt_ols_config.cluster_utils,
        "run_oc",
        _configmap_run_oc_stub(_olsconfig_configmap("openai", "gpt-6-luna"), calls),
    )

    adapt_ols_config.switch_default_provider("openai", "gpt-6-luna")

    assert [args for args, _ in calls] == [["get", "cm/olsconfig", "-o", "yaml"]]


def test_switch_default_provider_patches_configmap_and_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch the defaults in the configmap and bounce the app-server."""
    calls: list[tuple[list[str], str | None]] = []
    monkeypatch.setattr(
        adapt_ols_config.cluster_utils,
        "run_oc",
        _configmap_run_oc_stub(_olsconfig_configmap("openai", "gpt-6-luna"), calls),
    )

    adapt_ols_config.switch_default_provider("bedrock", "openai.gpt-6-luna")

    assert calls[0][0] == ["get", "cm/olsconfig", "-o", "yaml"]
    scale_calls = [args for args, _ in calls if args[0] == "scale"]
    assert scale_calls == [
        ["scale", "deployment/lightspeed-app-server", "--replicas", "0"],
        ["scale", "deployment/lightspeed-app-server", "--replicas", "1"],
    ]
    apply_calls = [args for args, _ in calls if args[:2] == ["apply", "-f"]]
    assert len(apply_calls) == 1
    applied_configmap = yaml.safe_load(
        next(command for args, command in calls if args[:2] == ["apply", "-f"])
    )
    applied = yaml.safe_load(applied_configmap["data"][DEFAULT_CONFIGURATION_FILE])
    assert applied["llm_config"]["default_provider"] == "bedrock"
    assert applied["llm_config"]["default_model"] == "openai.gpt-6-luna"


def test_apply_olsconfig_uses_combined_inspection_cr_for_multi_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load the combined inspection CR for the multi-provider inspection suite."""
    applied: dict[str, Any] = {}

    def fake_run_oc(
        args: list[str],
        command: str | None = None,
        ignore_existing_resource: bool = False,
    ) -> None:
        applied["args"] = args
        applied["config"] = yaml.safe_load(command)

    monkeypatch.setenv("OLS_CONFIG_SUFFIX", "mcp_inspection_all")
    monkeypatch.setattr(adapt_ols_config.cluster_utils, "run_oc", fake_run_oc)

    adapt_ols_config.apply_olsconfig(
        ["openai", "azure_openai", "bedrock", "google_vertex"]
    )

    spec = applied["config"]["spec"]
    provider_names = [provider["name"] for provider in spec["llm"]["providers"]]
    assert applied["args"] == ["apply", "-f", "-"]
    assert provider_names == [
        "openai",
        "azure_openai",
        "bedrock",
        "google_vertex",
        "google_vertex_anthropic",
        "watsonx",
        "rhoai_vllm",
        "rhelai_vllm",
    ]
    assert spec["ols"]["guardrails"]["toolResultInspection"]["enabled"] is True


def test_apply_olsconfig_substitutes_environment_for_lseval_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Substitute the model URL before applying the special provider template."""
    applied: dict[str, Any] = {}

    def fake_run_oc(
        args: list[str],
        command: str | None = None,
        ignore_existing_resource: bool = False,
    ) -> None:
        applied["args"] = args
        applied["config"] = yaml.safe_load(command)

    monkeypatch.setenv("OLS_CONFIG_SUFFIX", "default")
    monkeypatch.setenv("KSVC_URL", "https://model.example.test")
    monkeypatch.setattr(adapt_ols_config.cluster_utils, "run_oc", fake_run_oc)

    adapt_ols_config.apply_olsconfig(["rhoai_vllm_lseval"])

    provider = applied["config"]["spec"]["llm"]["providers"][0]
    assert applied["args"] == ["apply", "-f", "-"]
    assert provider["url"] == "https://model.example.test/v1"
    assert (
        applied["config"]["spec"]["ols"]["guardrails"]["toolResultInspection"][
            "enabled"
        ]
        is False
    )


def test_switch_default_provider_restores_deployment_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scale the app-server back up when a provider switch fails midway."""
    calls: list[tuple[list[str], str | None]] = []
    monkeypatch.setattr(
        adapt_ols_config.cluster_utils,
        "run_oc",
        _configmap_run_oc_stub(_olsconfig_configmap("openai", "gpt-6-luna"), calls),
    )
    monkeypatch.setattr(
        adapt_ols_config, "retry_until_timeout_or_success", lambda *_a, **_k: False
    )

    with pytest.raises(RuntimeError, match="did not terminate"):
        adapt_ols_config.switch_default_provider("bedrock", "openai.gpt-6-luna")

    scale_calls = [args for args, _ in calls if args[0] == "scale"]
    assert scale_calls == [
        ["scale", "deployment/lightspeed-app-server", "--replicas", "0"],
        ["scale", "deployment/lightspeed-app-server", "--replicas", "1"],
    ]
