"""Test e2e configuration changes through the cluster API."""

from copy import deepcopy
from io import StringIO
from types import ModuleType
from unittest.mock import Mock

import pytest
import yaml
from kubernetes.client import ApiException
from openshift.dynamic.exceptions import ConflictError, ForbiddenError

from ols.constants import DEFAULT_CONFIGURATION_FILE
from tests.e2e import conftest as e2e_conftest
from tests.e2e.utils import adapt_ols_config, cluster_api, ols_installer, olsconfig_cr
from tests.e2e.utils import cluster as cluster_utils


@pytest.mark.parametrize("providers", [["openai"], ["openai", "watsonx"]])
def test_apply_olsconfig_inspection_does_not_leak(
    monkeypatch: pytest.MonkeyPatch, providers: list[str]
) -> None:
    """Set inspection independently on each application of a template."""
    template = {
        "apiVersion": "ols.openshift.io/v1alpha1",
        "kind": "OLSConfig",
        "metadata": {"name": "cluster"},
        "spec": {"ols": {}},
    }
    monkeypatch.setattr(
        olsconfig_cr,
        "open",
        lambda *_a, **_k: StringIO(yaml.safe_dump(template)),
        raising=False,
    )
    create = Mock()
    monkeypatch.setattr(cluster_api, "create_resource", create)
    for suffix, expected in (
        ("mcp_inspection", True),
        ("mcp_inspection_gpt_6_luna", True),
        ("default", False),
        ("mcp_inspection", True),
        ("default", False),
    ):
        monkeypatch.setenv("OLS_CONFIG_SUFFIX", suffix)
        adapt_ols_config.apply_olsconfig(providers)
        assert (
            create.call_args.args[0]["spec"]["ols"]["guardrails"][
                "toolResultInspection"
            ]["enabled"]
            is expected
        )


@pytest.mark.parametrize(
    "suffix,expected",
    [
        ("mcp_inspection", ["inspection", "ordinary"]),
        ("mcp_inspection_gpt_6_luna", ["inspection", "ordinary"]),
        ("default", ["ordinary"]),
    ],
)
def test_collection_keeps_inspection_tests_only_for_inspection_suffix(
    monkeypatch: pytest.MonkeyPatch, suffix: str, expected: list[str]
) -> None:
    """Select inspection-marked tests only for the matching suite."""
    monkeypatch.setenv("OLS_CONFIG_SUFFIX", suffix)
    inspection, ordinary = Mock(), Mock()
    inspection.name, ordinary.name = "inspection", "ordinary"
    inspection.get_closest_marker.side_effect = lambda marker: (
        object() if marker == "inspection" else None
    )
    ordinary.get_closest_marker.return_value = None
    items = [inspection, ordinary]
    e2e_conftest.pytest_collection_modifyitems(items)
    assert [item.name for item in items] == expected


@pytest.mark.parametrize(
    "providers,suffix",
    [
        (["openai"], "mcp_inspection"),
        (
            [
                "openai",
                "azure_openai",
                "bedrock",
                "google_vertex",
                "google_vertex_anthropic",
                "watsonx",
                "rhoai_vllm",
                "rhelai_vllm",
            ],
            "mcp_inspection_all",
        ),
    ],
)
def test_apply_olsconfig_uses_inspection_template(
    monkeypatch: pytest.MonkeyPatch, providers: list[str], suffix: str
) -> None:
    """Apply the real single- or multi-provider inspection template."""
    create = Mock()
    monkeypatch.setattr(cluster_api, "create_resource", create)
    monkeypatch.setenv("OLS_CONFIG_SUFFIX", suffix)
    adapt_ols_config.apply_olsconfig(providers)
    spec = create.call_args.args[0]["spec"]
    assert spec["ols"]["guardrails"]["toolResultInspection"]["enabled"] is True
    assert spec["ols"]["toolsApprovalConfig"]["approvalType"] == "never"
    assert [provider["name"] for provider in spec["llm"]["providers"]] == providers


def test_apply_olsconfig_substitutes_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Substitute the model endpoint in the evaluation template."""
    create = Mock()
    monkeypatch.setattr(cluster_api, "create_resource", create)
    monkeypatch.setenv("OLS_CONFIG_SUFFIX", "default")
    monkeypatch.setenv("KSVC_URL", "https://model.example.test")
    adapt_ols_config.apply_olsconfig(["rhoai_vllm_lseval"])
    spec = create.call_args.args[0]["spec"]
    assert spec["llm"]["providers"][0]["url"] == "https://model.example.test/v1"
    assert spec["ols"]["guardrails"]["toolResultInspection"]["enabled"] is False


@pytest.mark.parametrize("scenario", ["unchanged", "changed", "failure"])
def test_switch_default_provider(
    monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    """Update defaults with a clean restart and restore replicas after failures."""
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "olsconfig"},
        "data": {
            DEFAULT_CONFIGURATION_FILE: yaml.safe_dump(
                {
                    "llm_config": {
                        "default_provider": "openai",
                        "default_model": "gpt-6-luna",
                    }
                }
            )
        },
    }
    read, apply, scale = Mock(return_value=deepcopy(configmap)), Mock(), Mock()
    monkeypatch.setattr(cluster_api, "read_resource", read)
    monkeypatch.setattr(cluster_api, "patch_resource", apply)
    monkeypatch.setattr(cluster_api, "scale_deployment", scale)
    monkeypatch.setattr(
        adapt_ols_config.cluster_utils, "get_running_pods", lambda _namespace: []
    )
    if scenario == "failure":
        monkeypatch.setattr(
            adapt_ols_config, "retry_until_timeout_or_success", lambda *_a, **_k: False
        )
        with pytest.raises(RuntimeError, match="did not terminate"):
            adapt_ols_config.switch_default_provider("bedrock", "openai.gpt-6-luna")
        apply.assert_not_called()
    elif scenario == "unchanged":
        adapt_ols_config.switch_default_provider("openai", "gpt-6-luna")
        apply.assert_not_called()
        scale.assert_not_called()
    else:
        adapt_ols_config.switch_default_provider("bedrock", "openai.gpt-6-luna")
        assert apply.call_args.args[:2] == ("configmap", "olsconfig")
        assert set(apply.call_args.args[2]) == {"data"}
        config = yaml.safe_load(
            apply.call_args.args[2]["data"][DEFAULT_CONFIGURATION_FILE]
        )
        assert config["llm_config"] == {
            "default_provider": "bedrock",
            "default_model": "openai.gpt-6-luna",
        }
    read.assert_called_once_with("configmap", "olsconfig")
    if scenario != "unchanged":
        assert [call.args for call in scale.call_args_list] == [
            ("lightspeed-app-server", 0),
            ("lightspeed-app-server", 1),
        ]


@pytest.mark.parametrize("providers", [["openai"], ["openai", "watsonx"]])
def test_apply_olsconfig_rejects_existing_configuration(
    monkeypatch: pytest.MonkeyPatch, providers: list[str]
) -> None:
    """Never claim success when an existing CR prevents the requested configuration."""
    monkeypatch.setenv("OLS_CONFIG_SUFFIX", "default")
    client = Mock()
    client.resources.get.return_value.create.side_effect = ConflictError(
        ApiException(status=409)
    )
    monkeypatch.setattr(cluster_api, "get_client", lambda: client)
    with pytest.raises(ConflictError):
        adapt_ols_config.apply_olsconfig(providers)


@pytest.mark.parametrize(
    "failure",
    [TimeoutError("finalizer stuck"), ForbiddenError(ApiException(status=403))],
)
def test_reconcile_stops_when_olsconfig_deletion_fails(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """Do not warm images or create a replacement CR after failed deletion."""
    scale = Mock()
    monkeypatch.setattr(cluster_api, "scale_deployment", scale)
    monkeypatch.setattr(
        adapt_ols_config, "retry_until_timeout_or_success", Mock(return_value=True)
    )
    monkeypatch.setattr(cluster_api, "delete_resource", Mock(side_effect=failure))
    warm, apply = Mock(), Mock()
    monkeypatch.setattr(adapt_ols_config, "warm_up_rhokp_image", warm)
    monkeypatch.setattr(adapt_ols_config, "apply_olsconfig", apply)
    with pytest.raises(type(failure)):
        adapt_ols_config._reconcile_olsconfig_with_operator(["openai", "watsonx"])
    warm.assert_not_called()
    apply.assert_not_called()
    assert [call.args for call in scale.call_args_list] == [
        ("lightspeed-operator-controller-manager", 1),
        ("lightspeed-operator-controller-manager", 0),
    ]


@pytest.mark.parametrize("module", [adapt_ols_config, ols_installer])
def test_configmap_updates_patch_only_configuration_data(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> None:
    """Avoid taking server-side ownership of a fetched operator-managed ConfigMap."""
    original = {
        "metadata": {"name": "olsconfig", "resourceVersion": "42"},
        "data": {
            "unrelated": "keep-me",
            DEFAULT_CONFIGURATION_FILE: yaml.safe_dump(
                {"ols_config": {"logging_config": {"lib_log_level": "DEBUG"}}}
            ),
        },
    }
    monkeypatch.setenv("OLS_CONFIG_SUFFIX", "default")
    monkeypatch.setattr(
        cluster_api, "read_resource", Mock(return_value=deepcopy(original))
    )
    patch = Mock()
    monkeypatch.setattr(cluster_api, "patch_resource", patch)
    monkeypatch.setattr(
        cluster_api,
        "apply_resource",
        Mock(side_effect=AssertionError("Unexpected apply")),
    )
    if module is adapt_ols_config:
        module.update_ols_configmap()
    else:
        module.update_ols_config()
    assert patch.call_args.args[:2] == ("configmap", "olsconfig")
    body = patch.call_args.args[2]
    assert set(body) == {"data"}
    assert set(body["data"]) == {DEFAULT_CONFIGURATION_FILE}
    config = yaml.safe_load(body["data"][DEFAULT_CONFIGURATION_FILE])
    assert config["ols_config"]["logging_config"]["lib_log_level"] == "INFO"


def test_disable_responses_api_targets_only_ollama_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Responses API is disabled only for ollama models via a data-only patch."""
    original = {
        "metadata": {"name": "olsconfig", "resourceVersion": "42"},
        "data": {
            "unrelated": "keep-me",
            DEFAULT_CONFIGURATION_FILE: yaml.safe_dump(
                {
                    "llm_providers": [
                        {
                            "name": "ollama",
                            "type": "openai",
                            "url": "http://ollama-service.ollama.svc.cluster.local:11434/v1",
                            "models": [{"name": "gemma3:1b"}],
                        },
                        {
                            "name": "openai",
                            "type": "openai",
                            "models": [{"name": "gpt-4o"}],
                        },
                    ],
                }
            ),
        },
    }
    monkeypatch.setattr(
        cluster_api, "read_resource", Mock(return_value=deepcopy(original))
    )
    patch = Mock()
    monkeypatch.setattr(cluster_api, "patch_resource", patch)
    monkeypatch.setattr(
        cluster_api,
        "apply_resource",
        Mock(side_effect=AssertionError("Unexpected apply")),
    )
    adapt_ols_config.disable_responses_api_for_ollama()
    assert patch.call_args.args[:2] == ("configmap", "olsconfig")
    body = patch.call_args.args[2]
    assert set(body) == {"data"}
    assert set(body["data"]) == {DEFAULT_CONFIGURATION_FILE}
    config = yaml.safe_load(body["data"][DEFAULT_CONFIGURATION_FILE])
    ollama_model = config["llm_providers"][0]["models"][0]
    assert ollama_model["name"] == "gemma3:1b"
    assert ollama_model["options"] == {"use_responses_api": False}
    # other providers keep the Responses API default
    other_model = config["llm_providers"][1]["models"][0]
    assert other_model == {"name": "gpt-4o"}


def test_disable_responses_api_fails_fast_without_ollama_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configmap without an ollama provider fails fast instead of no-op."""
    original = {
        "data": {
            DEFAULT_CONFIGURATION_FILE: yaml.safe_dump(
                {
                    "llm_providers": [
                        {"name": "openai", "type": "openai", "models": [{"name": "x"}]}
                    ]
                }
            )
        }
    }
    monkeypatch.setattr(cluster_api, "read_resource", Mock(return_value=original))
    patch = Mock()
    monkeypatch.setattr(cluster_api, "patch_resource", patch)
    with pytest.raises(RuntimeError, match="No 'ollama' provider found"):
        adapt_ols_config.disable_responses_api_for_ollama()
    patch.assert_not_called()


def test_disable_responses_api_fails_fast_on_missing_configmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing ConfigMap surfaces as a RuntimeError, not a raw KeyError."""
    monkeypatch.setattr(
        cluster_api,
        "read_resource",
        Mock(side_effect=ApiException(status=404, reason="Not Found")),
    )
    with pytest.raises(RuntimeError, match="Failed to disable Responses API"):
        adapt_ols_config.disable_responses_api_for_ollama()


def test_apply_e2e_specific_config_disables_responses_api_when_disconnected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disconnected runs patch the ConfigMap to disable the Responses API."""
    monkeypatch.setattr(adapt_ols_config, "disconnected", "true")
    monkeypatch.setattr(cluster_api, "scale_deployment", Mock(return_value=None))
    monkeypatch.setattr(
        adapt_ols_config,
        "retry_until_timeout_or_success",
        Mock(return_value=True),
    )
    disable = Mock()
    monkeypatch.setattr(adapt_ols_config, "disable_responses_api_for_ollama", disable)
    update = Mock()
    monkeypatch.setattr(adapt_ols_config, "update_ols_configmap", update)
    monkeypatch.setattr(adapt_ols_config, "wait_for_deployment", Mock())
    monkeypatch.setattr(cluster_utils, "deployment_exists", Mock(return_value=True))
    adapt_ols_config._apply_e2e_specific_config(ols_image="")
    disable.assert_called_once()
    update.assert_not_called()


def test_apply_e2e_specific_config_skips_responses_api_when_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connected runs keep the Responses API default for real providers."""
    monkeypatch.setattr(adapt_ols_config, "disconnected", "")
    monkeypatch.setattr(cluster_api, "scale_deployment", Mock(return_value=None))
    monkeypatch.setattr(
        adapt_ols_config,
        "retry_until_timeout_or_success",
        Mock(return_value=True),
    )
    disable = Mock()
    monkeypatch.setattr(adapt_ols_config, "disable_responses_api_for_ollama", disable)
    update = Mock()
    monkeypatch.setattr(adapt_ols_config, "update_ols_configmap", update)
    monkeypatch.setattr(adapt_ols_config, "wait_for_deployment", Mock())
    monkeypatch.setattr(cluster_utils, "deployment_exists", Mock(return_value=True))
    adapt_ols_config._apply_e2e_specific_config(ols_image="")
    disable.assert_not_called()
    update.assert_called_once()
