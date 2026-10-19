"""Validate provider-specific tool-result inspection OLSConfig fixtures."""

from pathlib import Path
from typing import Any

import pytest
import yaml

INSPECTION_CONFIGS = [
    (
        "openai",
        "mcp_inspection_ci",
        "openai",
        "gpt-6-luna",
    ),
    (
        "azure_openai",
        "mcp_inspection_ci",
        "azure_openai",
        "gpt-6-luna",
    ),
    (
        "bedrock_anthropic",
        "mcp_inspection_ci",
        "bedrock",
        "anthropic.claude-sonnet-4-6",
    ),
    (
        "bedrock_openai",
        "mcp_inspection_ci",
        "bedrock",
        "openai.gpt-6-luna",
    ),
    (
        "google_vertex",
        "mcp_inspection_ci",
        "google_vertex",
        "gemini-3.1-flash-lite",
    ),
    (
        "google_vertex_anthropic",
        "mcp_inspection_ci",
        "google_vertex_anthropic",
        "claude-sonnet-4-6",
    ),
    (
        "watsonx",
        "mcp_inspection_ci",
        "watsonx",
        "ibm/granite-4-h-small",
    ),
    (
        "rhoai_vllm",
        "mcp_inspection_ci",
        "rhoai_vllm",
        "gpt-4.1-mini",
    ),
    (
        "rhelai_vllm",
        "mcp_inspection_ci",
        "rhelai_vllm",
        "gpt-4.1-mini",
    ),
]


@pytest.mark.parametrize(
    ("provider_name", "suffix", "expected_type", "expected_model"),
    INSPECTION_CONFIGS,
    ids=[entry[0] for entry in INSPECTION_CONFIGS],
)
def test_inspection_config_matches_ci_provider_model(
    provider_name: str,
    suffix: str,
    expected_type: str,
    expected_model: str,
) -> None:
    """Match each fixture's configured provider and model to the CI matrix."""
    config_path = (
        Path("tests/config/operator_install")
        / f"olsconfig.crd.{provider_name}_{suffix}.yaml"
    )
    with config_path.open(encoding="utf-8") as config_file:
        config: dict[str, Any] = yaml.safe_load(config_file)

    spec = config["spec"]
    ols_config = spec["ols"]
    provider = spec["llm"]["providers"][0]

    assert provider["name"] == expected_type
    assert provider["type"] == expected_type
    assert provider["models"][0]["name"] == expected_model
    assert ols_config["defaultProvider"] == expected_type
    assert ols_config["defaultModel"] == expected_model
    assert ols_config["guardrails"]["toolResultInspection"]["enabled"] is True
    assert ols_config["toolsApprovalConfig"]["approvalType"] == "never"
    assert "byokRAGOnly" not in ols_config
    assert "mock-inspection" in {server["name"] for server in spec["mcpServers"]}
    assert {"MCPServer", "ToolFiltering"}.issubset(spec["featureGates"])
