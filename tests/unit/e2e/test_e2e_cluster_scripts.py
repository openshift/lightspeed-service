"""Unit tests for the cluster e2e suite selection."""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SUITE = re.compile(
    r'^\s*run_suite "(?P<name>[^"]+)" "(?P<tags>[^"]*)" .* '
    r'"\$OLS_IMAGE" "(?P<suffix>[^"]+)"\s*$'
)


@pytest.mark.parametrize(
    "script_name",
    ["test-e2e-cluster.sh", "test-e2e-cluster-periodics.sh"],
)
def test_cluster_script_has_one_gated_inspection_suite(script_name: str) -> None:
    """Select inspection tests only with the dedicated suffix and opt-in gate."""
    script = (REPO_ROOT / "tests/scripts" / script_name).read_text(encoding="utf-8")
    suites = [
        match.groupdict()
        for line in script.splitlines()
        if (match := RUN_SUITE.match(line)) is not None
    ]
    inspection_suites = [suite for suite in suites if suite["tags"] == "inspection"]
    dedicated_suffixes = [
        suite for suite in suites if suite["suffix"] == "mcp_inspection"
    ]

    assert len(inspection_suites) == 1
    assert inspection_suites == dedicated_suffixes
    assert "RUN_TOOL_RESULT_INSPECTION_E2E" in script
    assert re.search(
        r'if \[\[ "\$\{RUN_TOOL_RESULT_INSPECTION_E2E:-[01]\}" == "1" \]\]; then\s+'
        r'run_suite "[^"]+" "inspection"',
        script,
    )


@pytest.mark.parametrize(
    ("flag", "inspection_runs"), [(None, True), ("0", False), ("1", True)]
)
def test_pr_cluster_script_selects_inspection_suite(
    flag: str | None, inspection_runs: bool
) -> None:
    """The PR matrix isolates inspection while honoring an explicit override."""
    script = (REPO_ROOT / "tests/scripts/test-e2e-cluster.sh").read_text(
        encoding="utf-8"
    )
    body = script.split("function run_suites() {", 1)[1].split(
        "\nfunction finish()", 1
    )[0]
    shell = (
        'run_suite() { printf "%s:%s\\n" "$1" "$7"; }\n'
        "cleanup_ols_operator() { :; }\n"
        f"function run_suites() {{{body}\n"
        "run_suites\n"
    )
    env = os.environ.copy()
    env.pop("RUN_TOOL_RESULT_INSPECTION_E2E", None)
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
    suites = result.stdout.splitlines()
    assert (
        suites.count("openai_tool_result_inspection:mcp_inspection") == inspection_runs
    )
    assert "openai:default" in suites
    assert "openai_mcp:mcp" in suites
    assert all(
        suffix != "mcp_inspection"
        for suite, suffix in (entry.split(":", 1) for entry in suites)
        if suite != "openai_tool_result_inspection"
    )
