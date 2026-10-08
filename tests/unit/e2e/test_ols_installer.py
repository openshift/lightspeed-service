"""Unit tests for e2e OLS installation."""

from typing import Any

import pytest
import yaml

from tests.e2e.utils import ols_installer


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
