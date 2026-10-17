"""Unit tests for the cluster e2e suite selection."""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INSPECTION_RUN_SUITE = re.compile(
    r'^\s*run_suite "(?P<name>[^"]+)" "inspection" '
    r'"(?P<provider>[^"]+)" "(?P<credentials>[^"]+)" '
    r'"(?P<model>[^"]+)" "\$OLS_IMAGE" "(?P<suffix>[^"]+)"\s*$'
)
PR_INSPECTION_TARGETS = {
    (
        "openai_gpt6_tool_result_inspection",
        "openai",
        "$OPENAI_PROVIDER_KEY_PATH",
        "gpt-6-luna",
        "mcp_inspection_ci",
    ),
    (
        "azure_openai_tool_result_inspection",
        "azure_openai",
        "$AZUREOPENAI_PROVIDER_KEY_PATH",
        "gpt-6-luna",
        "mcp_inspection_ci",
    ),
    (
        "bedrock_anthropic_tool_result_inspection",
        "bedrock_anthropic",
        "iam",
        "anthropic.claude-sonnet-4-6",
        "mcp_inspection_ci",
    ),
    (
        "bedrock_openai_tool_result_inspection",
        "bedrock_openai",
        "iam",
        "openai.gpt-6-luna",
        "mcp_inspection_ci",
    ),
    (
        "google_vertex_tool_result_inspection",
        "google_vertex",
        "$VERTEX_PROVIDER_KEY_PATH",
        "gemini-3.1-flash-lite",
        "mcp_inspection_ci",
    ),
    (
        "google_vertex_anthropic_tool_result_inspection",
        "google_vertex_anthropic",
        "$VERTEX_PROVIDER_KEY_PATH",
        "claude-sonnet-4-6",
        "mcp_inspection_ci",
    ),
    (
        "watsonx_tool_result_inspection",
        "watsonx",
        "$WATSONX_PROVIDER_KEY_PATH",
        "ibm/granite-4-h-small",
        "mcp_inspection_ci",
    ),
    (
        "rhoai_vllm_tool_result_inspection",
        "rhoai_vllm",
        "$OPENAI_PROVIDER_KEY_PATH",
        "gpt-4.1-mini",
        "mcp_inspection_ci",
    ),
    (
        "rhelai_vllm_tool_result_inspection",
        "rhelai_vllm",
        "$OPENAI_PROVIDER_KEY_PATH",
        "gpt-4.1-mini",
        "mcp_inspection_ci",
    ),
}


@pytest.mark.parametrize(
    ("script_name", "expected_targets"),
    [
        ("test-e2e-cluster.sh", PR_INSPECTION_TARGETS),
        ("test-e2e-cluster-periodics.sh", PR_INSPECTION_TARGETS),
    ],
)
def test_cluster_script_gates_expected_inspection_suites(
    script_name: str, expected_targets: set[tuple[str, str, str, str, str]]
) -> None:
    """Gate the audited provider inspection suites behind the existing flag."""
    script = (REPO_ROOT / "tests/scripts" / script_name).read_text(encoding="utf-8")
    inspection_targets = [
        match.groups()
        for line in script.splitlines()
        if (match := INSPECTION_RUN_SUITE.match(line)) is not None
    ]

    assert len(inspection_targets) == len(expected_targets)
    assert set(inspection_targets) == expected_targets
    assert "RUN_TOOL_RESULT_INSPECTION_E2E" in script
    assert re.search(
        r'if \[\[ "\$\{RUN_TOOL_RESULT_INSPECTION_E2E:-[01]\}" == "1" \]\]; then\s+'
        r'run_suite "[^"]+" "inspection"',
        script,
    )


@pytest.mark.parametrize(
    ("script_name", "flag", "expected_targets"),
    [
        ("test-e2e-cluster.sh", None, PR_INSPECTION_TARGETS),
        ("test-e2e-cluster.sh", "0", set()),
        ("test-e2e-cluster.sh", "1", PR_INSPECTION_TARGETS),
        ("test-e2e-cluster-periodics.sh", None, set()),
        ("test-e2e-cluster-periodics.sh", "0", set()),
        ("test-e2e-cluster-periodics.sh", "1", PR_INSPECTION_TARGETS),
    ],
)
def test_cluster_script_selects_inspection_matrix(
    script_name: str,
    flag: str | None,
    expected_targets: set[tuple[str, str, str, str, str]],
) -> None:
    """Run the expected inspection matrix and honor each script's default gate."""
    script = (REPO_ROOT / "tests/scripts" / script_name).read_text(encoding="utf-8")
    body = script.split("function run_suites() {", 1)[1].split(
        "\nfunction finish()", 1
    )[0]
    shell = (
        'run_suite() { printf "%s:%s:%s:%s\\n" "$1" "$3" "$5" "$7"; }\n'
        "cleanup_ols_operator() { :; }\n"
        f"function run_suites() {{{body}\n"
        "run_suites\n"
    )
    env = os.environ.copy()
    env.pop("RUN_TOOL_RESULT_INSPECTION_E2E", None)
    env["DISCONNECTED"] = ""
    if flag is not None:
        env["RUN_TOOL_RESULT_INSPECTION_E2E"] = flag
    env.update(
        dict.fromkeys(
            (
                "AZUREOPENAI_PROVIDER_KEY_PATH",
                "OPENAI_PROVIDER_KEY_PATH",
                "VERTEX_PROVIDER_KEY_PATH",
                "WATSONX_PROVIDER_KEY_PATH",
                "OLS_IMAGE",
            ),
            "unused",
        )
    )
    result = subprocess.run(  # noqa: S603
        ["bash", "-c", shell],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    suites = [tuple(line.split(":")) for line in result.stdout.splitlines()]
    inspection_suites = [
        suite for suite in suites if suite[3].startswith("mcp_inspection")
    ]
    expected_runs = {
        (name, provider, model, suffix)
        for name, provider, _credentials, model, suffix in expected_targets
    }

    assert len(inspection_suites) == len(expected_runs)
    assert set(inspection_suites) == expected_runs
    assert ("openai", "openai", "gpt-6-luna", "default") in suites
