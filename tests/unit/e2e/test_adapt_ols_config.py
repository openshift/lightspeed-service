"""Unit tests for e2e OLS configuration adaptation."""

from io import StringIO
from typing import Any

import pytest
import yaml

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
