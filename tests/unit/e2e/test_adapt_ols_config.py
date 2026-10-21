"""Unit tests for e2e OLS configuration adaptation."""

from io import StringIO
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

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
