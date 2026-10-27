"""Validate the combined tool-calling OLSConfig fixtures."""

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "tests" / "config" / "operator_install"

TOOL_CALLING_MATRIX_RUN_SUITE = re.compile(
    r'^\s*run_suite "[^"]+" "(?P<tags>tool_calling|gpt6_tool_calling)" '
    r'"(?P<entries>[^"]+)" "[^"]+" "[^"]+" "\$OLS_IMAGE" "(?P<suffix>[^"]+)"\s*$'
)


def _load_config(name: str) -> dict[str, Any]:
    """Load a combined tool-calling OLSConfig CR."""
    with (CONFIG_DIR / f"olsconfig.crd.{name}.yaml").open(encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def _providers_by_name(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the CR provider entries by provider name."""
    return {
        provider["name"]: provider for provider in config["spec"]["llm"]["providers"]
    }


def _matrix_entries(script_name: str, tags: str) -> set[tuple[str, str]]:
    """Collect provider/model pairs from a script's matrix run_suite lines."""
    script = (REPO_ROOT / "tests" / "scripts" / script_name).read_text(encoding="utf-8")
    entries = set()
    for line in script.splitlines():
        match = TOOL_CALLING_MATRIX_RUN_SUITE.match(line)
        if match is not None and match.group("tags") == tags:
            for entry in match.group("entries").split():
                _suite, provider, model = entry.split(":")
                entries.add((provider, model))
    return entries


def test_tool_calling_all_configures_the_full_provider_matrix() -> None:
    """Cover every provider/model pair the cluster scripts select."""
    config = _load_config("tool_calling_all")
    providers = _providers_by_name(config)

    script_entries = _matrix_entries("test-e2e-cluster.sh", "tool_calling") | (
        _matrix_entries("test-e2e-cluster-periodics.sh", "tool_calling")
    )
    for provider_name, model in script_entries:
        assert (
            provider_name in providers
        ), f"provider {provider_name} is missing from tool_calling_all"
        models = [entry["name"] for entry in providers[provider_name]["models"]]
        assert model in models, f"model {model} is missing for provider {provider_name}"


def test_gpt6_tool_calling_all_configures_the_gpt6_matrix() -> None:
    """Cover every provider/model pair the gpt6 cluster scripts select."""
    config = _load_config("gpt6_tool_calling_all")
    providers = _providers_by_name(config)

    script_entries = _matrix_entries("test-e2e-cluster.sh", "gpt6_tool_calling")
    for provider_name, model in script_entries:
        assert provider_name in providers
        models = [entry["name"] for entry in providers[provider_name]["models"]]
        assert model in models


def test_tool_calling_all_provider_shapes() -> None:
    """Match provider types, models, secrets, and reasoning configuration."""
    config = _load_config("tool_calling_all")
    providers = _providers_by_name(config)

    assert set(providers) == {
        "openai",
        "azure_openai",
        "bedrock",
        "google_vertex_anthropic",
        "google_vertex",
        "watsonx",
    }
    expected_models = {
        "openai": ["gpt-6-luna"],
        "azure_openai": ["gpt-6-luna"],
        "bedrock": ["openai.gpt-6-luna"],
        "google_vertex_anthropic": ["claude-sonnet-4-6"],
        "google_vertex": ["gemini-3.1-flash-lite"],
        "watsonx": ["ibm/granite-4-h-small"],
    }
    expected_secrets = {
        "openai": "openaicreds",
        "azure_openai": "azure-openaicreds",
        "bedrock": "llmcreds",
        "google_vertex_anthropic": "google-vertex-anthropiccreds",
        "google_vertex": "google-vertexcreds",
        "watsonx": "watsonxcreds",
    }
    for name, provider in providers.items():
        assert [entry["name"] for entry in provider["models"]] == expected_models[name]
        assert provider["credentialsSecretRef"]["name"] == expected_secrets[name]

    for name in ("openai", "azure_openai"):
        parameters = providers[name]["models"][0].get("parameters", {})
        assert (
            parameters.get("reasoningConfig", {}).get("effort") == "medium"
        ), f"{name} must keep its reasoning config in tool_calling_all"
    for name in ("bedrock", "google_vertex_anthropic", "google_vertex", "watsonx"):
        assert "parameters" not in providers[name]["models"][0]


def test_gpt6_tool_calling_all_has_no_reasoning_config() -> None:
    """Run the gpt6 matrix without reasoning config, as before consolidation."""
    config = _load_config("gpt6_tool_calling_all")
    providers = _providers_by_name(config)

    assert set(providers) == {"openai", "azure_openai"}
    for name, provider in providers.items():
        assert (
            "parameters" not in provider["models"][0]
        ), f"{name} must not carry reasoning config in gpt6_tool_calling_all"


def test_tool_calling_configs_enable_introspection_and_filters() -> None:
    """Keep the tool-calling ols section identical to the old per-provider CRs."""
    for name in ("tool_calling_all", "gpt6_tool_calling_all"):
        ols_config = _load_config(name)["spec"]["ols"]
        assert ols_config["defaultProvider"] == "openai"
        assert ols_config["defaultModel"] == "gpt-6-luna"
        assert ols_config["introspectionEnabled"] is True
        assert ols_config["logLevel"] == "DEBUG"
        assert {f["name"] for f in ols_config["queryFilters"]} == {
            "foo_filter",
            "bar_filter",
        }
        assert ols_config["userDataCollection"]["feedbackDisabled"] is True
        assert ols_config["userDataCollection"]["transcriptsDisabled"] is True


def test_pr_ci_bootstrap_covers_combined_cr_providers() -> None:
    """Provision credentials for every provider the combined CR configures.

    The PR-CI matrix tests fewer providers than the combined CR configures
    (watsonx tool-calling is periodic-only), but the CR cannot reconcile when
    a configured provider lacks its secret, so the bootstrap list must cover
    the whole CR regardless of the tested entries.
    """
    script = (REPO_ROOT / "tests" / "scripts" / "test-e2e-cluster.sh").read_text(
        encoding="utf-8"
    )
    providers_match = re.search(r'E2E_MATRIX_CREDENTIAL_PROVIDERS="([^"]+)"', script)
    key_paths_match = re.search(r'E2E_MATRIX_CREDENTIAL_KEY_PATHS="([^"]+)"', script)
    assert providers_match is not None, "bootstrap provider list is missing"
    assert key_paths_match is not None, "bootstrap key path list is missing"

    bootstrap_providers = providers_match.group(1).split()
    bootstrap_key_paths = key_paths_match.group(1).split()
    assert len(bootstrap_providers) == len(
        bootstrap_key_paths
    ), "bootstrap providers and key paths are unaligned"

    config = _load_config("tool_calling_all")
    cr_providers = {provider["name"] for provider in config["spec"]["llm"]["providers"]}
    assert cr_providers <= set(bootstrap_providers), (
        f"providers configured in the CR but not bootstrapped: "
        f"{cr_providers - set(bootstrap_providers)}"
    )

    tested_entries = _matrix_entries("test-e2e-cluster.sh", "tool_calling")
    assert {provider for provider, _model in tested_entries} <= set(
        bootstrap_providers
    ), "tested matrix entries must be covered by the bootstrap credentials"
