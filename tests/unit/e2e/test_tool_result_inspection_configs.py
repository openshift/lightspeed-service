"""Validate the combined tool-result inspection OLSConfig fixture."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.e2e.test_tool_result_inspection import DEFAULT_INSPECTION_MATRIX

COMBINED_CONFIG_PATH = (
    Path("tests/config/operator_install") / "olsconfig.crd.mcp_inspection_all.yaml"
)

# Provider entries expected in the combined CR. Secret names follow the
# create_secrets() multi-provider scheme (<provider> with "_" mapped to "-"
# plus "creds"); bedrock keeps the fixed llmcreds IAM secret.
EXPECTED_PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "type": "openai",
        "models": ["gpt-6-luna"],
        "secret": "openaicreds",
    },
    "azure_openai": {
        "type": "azure_openai",
        "models": ["gpt-6-luna"],
        "secret": "azure-openaicreds",
    },
    "bedrock": {
        "type": "bedrock",
        "models": ["anthropic.claude-sonnet-4-6", "openai.gpt-6-luna"],
        "secret": "llmcreds",
    },
    "google_vertex": {
        "type": "google_vertex",
        "models": ["gemini-3.1-flash-lite"],
        "secret": "google-vertexcreds",
    },
    "google_vertex_anthropic": {
        "type": "google_vertex_anthropic",
        "models": ["claude-sonnet-4-6"],
        "secret": "google-vertex-anthropiccreds",
    },
    "watsonx": {
        "type": "watsonx",
        "models": ["ibm/granite-4-h-small"],
        "secret": "watsonxcreds",
    },
    "rhoai_vllm": {
        "type": "rhoai_vllm",
        "models": ["gpt-4.1-mini"],
        "secret": "rhoai-vllmcreds",
    },
    "rhelai_vllm": {
        "type": "rhelai_vllm",
        "models": ["gpt-4.1-mini"],
        "secret": "rhelai-vllmcreds",
    },
}


def _load_combined_config() -> dict[str, Any]:
    """Load the combined inspection OLSConfig CR."""
    with COMBINED_CONFIG_PATH.open(encoding="utf-8") as config_file:
        config: dict[str, Any] = yaml.safe_load(config_file)
    return config


def _providers_by_name(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the CR provider entries by provider name."""
    return {
        provider["name"]: provider for provider in config["spec"]["llm"]["providers"]
    }


def test_combined_inspection_config_covers_ci_matrix() -> None:
    """Configure every provider/model pair the inspection tests select."""
    providers = _providers_by_name(_load_combined_config())

    for provider_name, model in DEFAULT_INSPECTION_MATRIX:
        assert (
            provider_name in providers
        ), f"provider {provider_name} is missing from the combined CR"
        models = [entry["name"] for entry in providers[provider_name]["models"]]
        assert model in models, f"model {model} is missing for provider {provider_name}"


@pytest.mark.parametrize("provider_name", sorted(EXPECTED_PROVIDERS))
def test_combined_inspection_config_provider_shape(provider_name: str) -> None:
    """Match each provider entry to its type, models, and credential secret."""
    expected = EXPECTED_PROVIDERS[provider_name]
    providers = _providers_by_name(_load_combined_config())

    assert provider_name in providers
    provider = providers[provider_name]
    assert provider["type"] == expected["type"]
    assert [entry["name"] for entry in provider["models"]] == expected["models"]
    assert provider["credentialsSecretRef"]["name"] == expected["secret"]


def test_combined_inspection_config_enables_inspection_suite() -> None:
    """Enable tool-result inspection and mock MCP servers for the suite."""
    config = _load_combined_config()
    spec = config["spec"]
    ols_config = spec["ols"]

    assert ols_config["defaultProvider"] == "openai"
    assert ols_config["defaultModel"] == "gpt-6-luna"
    assert ols_config["guardrails"]["toolResultInspection"]["enabled"] is True
    assert ols_config["toolsApprovalConfig"]["approvalType"] == "never"
    assert "byokRAGOnly" not in ols_config
    assert "mock-inspection" in {server["name"] for server in spec["mcpServers"]}
    assert {"MCPServer", "ToolFiltering"}.issubset(spec["featureGates"])
