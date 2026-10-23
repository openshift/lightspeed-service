"""Unit tests for the cluster e2e suite selection."""

import os
import re
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("external_mirrors", [None, "false", "true"])
def test_operator_cleanup_respects_external_mirrors(external_mirrors):
    """Do not delete mirrors owned by the HyperShift management cluster."""
    env = os.environ.copy()
    env.pop("OLS_EXTERNAL_IMAGE_MIRRORS", None)
    if external_mirrors is not None:
        env["OLS_EXTERNAL_IMAGE_MIRRORS"] = external_mirrors
    shell = (
        'oc() { printf "oc %s\\n" "$*"; }\n'
        "operator-sdk() { :; }\n"
        "source tests/scripts/utils.sh\n"
        "cleanup_ols_operator\n"
    )
    result = subprocess.run(  # noqa: S603
        ["bash", "-c", shell],  # noqa: S607
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert ("imagedigestmirrorset/" in result.stdout) == (external_mirrors != "true")
    assert "ns openshift-lightspeed" in result.stdout


REPO_ROOT = Path(__file__).resolve().parents[3]
INSPECTION_RUN_SUITE = re.compile(
    r'^\s*run_suite "(?P<name>[^"]+)" "inspection" '
    r'"(?P<provider>[^"]+)" "(?P<credentials>[^"]+)" '
    r'"(?P<model>[^"]+)" "\$OLS_IMAGE" "(?P<suffix>[^"]+)"\s*$'
)
PR_INSPECTION_TARGETS = {
    (
        "tool_result_inspection",
        "openai azure_openai bedrock google_vertex google_vertex_anthropic "
        "watsonx rhoai_vllm rhelai_vllm",
        "$OPENAI_PROVIDER_KEY_PATH $AZUREOPENAI_PROVIDER_KEY_PATH iam "
        "$VERTEX_PROVIDER_KEY_PATH $VERTEX_PROVIDER_KEY_PATH "
        "$WATSONX_PROVIDER_KEY_PATH $OPENAI_PROVIDER_KEY_PATH "
        "$OPENAI_PROVIDER_KEY_PATH",
        "multi",
        "mcp_inspection_all",
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
