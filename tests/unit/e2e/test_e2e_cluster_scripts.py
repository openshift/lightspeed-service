"""Unit tests for the cluster e2e suite selection."""

import json
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
SCRIPTS = REPO_ROOT / "tests" / "scripts"
INSPECTION_RUN_SUITE = re.compile(
    r'^\s*run_suite "(?P<name>[^"]+)" "inspection" '
    r'"(?P<provider>[^"]+)" "(?P<credentials>[^"]+)" '
    r'"(?P<model>[^"]+)" "\$OLS_IMAGE" "(?P<suffix>[^"]+)"\s*$'
)
TOOL_CALLING_MATRIX_RUN_SUITE = re.compile(
    r'^\s*run_suite "(?P<name>[^"]+)" "(?P<tags>tool_calling|gpt6_tool_calling)" '
    r'"(?P<entries>[^"]+)" "(?P<credentials>[^"]+)" '
    r'"(?P<model>[^"]+)" "\$OLS_IMAGE" "(?P<suffix>[^"]+)"\s*$'
)
PR_TOOL_CALLING_MATRICES = {
    (
        "tool_calling_matrix",
        "tool_calling",
        "openai_tool_calling:openai:gpt-6-luna "
        "azure_openai_tool_calling:azure_openai:gpt-6-luna "
        "bedrock_openai_tool_calling:bedrock:openai.gpt-6-luna "
        "google_vertex_anthropic_tool_calling:google_vertex_anthropic:claude-haiku-5-5 "
        "google_vertex_tool_calling:google_vertex:gemini-3.1-flash-lite",
        "$OPENAI_PROVIDER_KEY_PATH $AZUREOPENAI_PROVIDER_KEY_PATH iam "
        "$VERTEX_PROVIDER_KEY_PATH $VERTEX_PROVIDER_KEY_PATH",
        "multi",
        "tool_calling_all",
    ),
    (
        "gpt6_tool_calling_matrix",
        "gpt6_tool_calling",
        "openai_gpt6_tool_calling:openai:gpt-6-luna "
        "azure_openai_gpt6_tool_calling:azure_openai:gpt-6-luna",
        "$OPENAI_PROVIDER_KEY_PATH $AZUREOPENAI_PROVIDER_KEY_PATH",
        "multi",
        "gpt6_tool_calling_all",
    ),
}
PERIODIC_TOOL_CALLING_MATRICES = {
    (
        "tool_calling_matrix",
        "tool_calling",
        "openai_tool_calling:openai:gpt-6-luna "
        "azure_openai_tool_calling:azure_openai:gpt-6-luna "
        "bedrock_openai_tool_calling:bedrock:openai.gpt-6-luna "
        "google_vertex_anthropic_tool_calling:google_vertex_anthropic:claude-haiku-5-5 "
        "watsonx_tool_calling:watsonx:ibm/granite-4-h-small "
        "google_vertex_tool_calling:google_vertex:gemini-3.1-flash-lite",
        "$OPENAI_PROVIDER_KEY_PATH $AZUREOPENAI_PROVIDER_KEY_PATH iam "
        "$VERTEX_PROVIDER_KEY_PATH $WATSONX_PROVIDER_KEY_PATH "
        "$VERTEX_PROVIDER_KEY_PATH",
        "multi",
        "tool_calling_all",
    ),
    (
        "gpt6_tool_calling_matrix",
        "gpt6_tool_calling",
        "openai_gpt6_tool_calling:openai:gpt-6-luna "
        "azure_openai_gpt6_tool_calling:azure_openai:gpt-6-luna",
        "$OPENAI_PROVIDER_KEY_PATH $AZUREOPENAI_PROVIDER_KEY_PATH",
        "multi",
        "gpt6_tool_calling_all",
    ),
}
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
        "check_matrix_filter_matches() { :; }\n"
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


@pytest.mark.parametrize(
    ("script_name", "expected_matrices"),
    [
        ("test-e2e-cluster.sh", PR_TOOL_CALLING_MATRICES),
        ("test-e2e-cluster-periodics.sh", PERIODIC_TOOL_CALLING_MATRICES),
    ],
)
def test_cluster_script_runs_tool_calling_matrices(
    script_name: str, expected_matrices: set[tuple[str, str, str, str, str, str]]
) -> None:
    """Run the tool-calling provider matrices with paired key paths."""
    script = (REPO_ROOT / "tests" / "scripts" / script_name).read_text(encoding="utf-8")
    matrices = {
        match.groups()
        for line in script.splitlines()
        if (match := TOOL_CALLING_MATRIX_RUN_SUITE.match(line)) is not None
    }

    assert matrices == expected_matrices
    for _name, _tags, entries, credentials, _model, _suffix in expected_matrices:
        entry_list = entries.split()
        credential_list = credentials.split()
        assert len(entry_list) == len(
            credential_list
        ), f"matrix entries and key paths are unaligned: {entries!r}"
        for entry in entry_list:
            assert (
                len(entry.split(":")) == 3
            ), f"matrix entry is not suite:provider:model: {entry!r}"
        # the first entry must match the combined CR's default provider so
        # the first iteration needs no provider switch
        first_provider = entry_list[0].split(":")[1]
        assert (
            first_provider == "openai"
        ), f"matrix must start with the CR default provider: {entry_list[0]!r}"


def test_run_suite_matrix_mode_switches_provider_per_iteration(
    tmp_path: Path,
) -> None:
    """Reuse one deployment and switch the default provider per entry."""
    calls = tmp_path / "calls"
    matrix_entries = (
        "openai_tool_calling:openai:gpt-6-luna "
        "bedrock_openai_tool_calling:bedrock:openai.gpt-6-luna"
    )
    script = f"""
    source '{SCRIPTS / "utils.sh"}'
    matrix_entries="{matrix_entries}"
    make() {{
      printf '%s|%s|%s|%s|%s\\n' "$SUITE_ID" "$PROVIDER" "$PROVIDER_KEY_PATH" \
        "$SWITCH_DEFAULT_PROVIDER" "$SKIP_CLUSTER_SETUP" >> '{calls}'
    }}
    run_suite tool_calling_matrix "tool_calling" \\
      "$matrix_entries" "/keys/openai iam" multi image tool_calling_all
    """
    env = {**os.environ, "ARTIFACT_DIR": str(tmp_path)}
    subprocess.run(["/bin/bash", "-c", script], env=env, check=True)  # noqa: S603

    assert calls.read_text().splitlines() == [
        "openai_tool_calling|openai bedrock|/keys/openai iam|openai gpt-6-luna|false",
        "bedrock_openai_tool_calling|bedrock|iam|bedrock openai.gpt-6-luna|true",
    ]
    results = [
        json.loads(line)
        for line in (tmp_path / "suite_results.jsonl").read_text().splitlines()
    ]
    assert [result["suite"] for result in results] == [
        "openai_tool_calling",
        "bedrock_openai_tool_calling",
    ]
    assert all(result["rc"] == 0 for result in results)


def test_run_suite_matrix_mode_propagates_iteration_failures(
    tmp_path: Path,
) -> None:
    """Keep the first failing iteration's exit code for the matrix."""
    matrix_entries = (
        "openai_tool_calling:openai:gpt-6-luna "
        "bedrock_openai_tool_calling:bedrock:openai.gpt-6-luna"
    )
    script = f"""
    source '{SCRIPTS / "utils.sh"}'
    matrix_entries="{matrix_entries}"
    make() {{
      if [ "$SUITE_ID" = "bedrock_openai_tool_calling" ]; then return 3; fi
      return 0
    }}
    run_suite tool_calling_matrix "tool_calling" \\
      "$matrix_entries" "/keys/openai iam" multi image tool_calling_all
    echo "matrix-rc=$?"
    """
    env = {**os.environ, "ARTIFACT_DIR": str(tmp_path)}
    result = subprocess.run(  # noqa: S603
        ["bash", "-c", script],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "matrix-rc=3" in result.stdout
    results = [
        json.loads(line)
        for line in (tmp_path / "suite_results.jsonl").read_text().splitlines()
    ]
    assert [result["rc"] for result in results] == [0, 3]


def test_run_suite_matrix_mode_rejects_unaligned_key_paths(
    tmp_path: Path,
) -> None:
    """Fail fast when entries and key paths do not pair up."""
    script = f"""
    source '{SCRIPTS / "utils.sh"}'
    make() {{ :; }}
    run_suite bad_matrix "tool_calling" "a-suite:a:m b-suite:b:m" "only-one" \\
      multi image tool_calling_all
    echo "matrix-rc=$?"
    """
    env = {**os.environ, "ARTIFACT_DIR": str(tmp_path)}
    result = subprocess.run(  # noqa: S603
        ["bash", "-c", script],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "matrix entries but 1 key paths" in result.stdout
    assert "matrix-rc=2" in result.stdout


def test_run_suite_gathers_on_success_for_inspection_suite(tmp_path: Path) -> None:
    """Keep must-gather artifacts for the inspection suite's provider split."""
    calls = tmp_path / "calls"
    script = f"""
    source '{SCRIPTS / "utils.sh"}'
    make() {{
      printf '%s|%s\\n' "$SUITE_ID" "$MUST_GATHER_ON_SUCCESS" >> '{calls}'
    }}
    run_suite tool_result_inspection "inspection" "openai watsonx" "/k1 /k2" \\
      multi image mcp_inspection_all
    run_suite openai "not tool_calling" "openai" "/k1" "gpt-6-luna" image default
    """
    env = {**os.environ, "ARTIFACT_DIR": str(tmp_path)}
    subprocess.run(["/bin/bash", "-c", script], env=env, check=True)  # noqa: S603

    assert calls.read_text().splitlines() == [
        "tool_result_inspection|true",
        "openai|false",
    ]


def test_run_suite_matrix_filter_reruns_selected_provider(tmp_path: Path) -> None:
    """Rerun a single provider's matrix entry with the full setup intact."""
    calls = tmp_path / "calls"
    script = f"""
    source '{SCRIPTS / "utils.sh"}'
    make() {{
      printf '%s|%s|%s|%s\\n' "$SUITE_ID" "$PROVIDER" "$PROVIDER_KEY_PATH" \
        "$SWITCH_DEFAULT_PROVIDER" >> '{calls}'
    }}
    export E2E_MATRIX_FILTER="azure_tc"
    run_suite tool_calling_matrix "tool_calling" \\
      "openai_tc:openai:gpt-6-luna azure_tc:azure_openai:gpt-6-luna bedrock_tc:bedrock:m" \\
      "/keys/openai /keys/azure iam" multi image tool_calling_all
    """
    env = {**os.environ, "ARTIFACT_DIR": str(tmp_path)}
    env.pop("E2E_MATRIX_FILTER", None)
    subprocess.run(["/bin/bash", "-c", script], env=env, check=True)  # noqa: S603

    lines = calls.read_text().splitlines()
    assert lines == [
        # single filtered entry: full provider list for secrets/CR apply,
        # switch to the selected provider, its own key path
        "azure_tc|openai azure_openai bedrock|/keys/openai /keys/azure iam|azure_openai gpt-6-luna",
    ]
    assert (tmp_path / ".matrix_filter_matched").read_text() == "azure_tc\n"


def test_run_suite_matrix_filter_skips_matrix_without_matches(
    tmp_path: Path,
) -> None:
    """Skip a matrix whose entries are all filtered out.

    The other matrix may still match; the union is validated by
    check_matrix_filter_matches.
    """
    calls = tmp_path / "calls"
    script = f"""
    source '{SCRIPTS / "utils.sh"}'
    make() {{ printf '%s\\n' "$SUITE_ID" >> '{calls}'; }}
    export E2E_MATRIX_FILTER="azure_openai_gpt6_tool_calling"
    run_suite tool_calling_matrix "tool_calling" "a-suite:a:m b-suite:b:m" \\
      "/k1 /k2" multi image tool_calling_all
    echo "matrix-rc=$?"
    """
    env = {**os.environ, "ARTIFACT_DIR": str(tmp_path)}
    result = subprocess.run(  # noqa: S603
        ["bash", "-c", script],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "matched no entries of suite tool_calling_matrix - skipping" in (
        result.stderr
    )
    assert "matrix-rc=0" in result.stdout
    assert not calls.exists() or calls.read_text() == ""


def test_check_matrix_filter_matches_validates_the_union(tmp_path: Path) -> None:
    """Reject a filter that matched no entry in any matrix; accept matches."""
    script = f"""
    source '{SCRIPTS / "utils.sh"}'
    unset E2E_MATRIX_FILTER
    check_matrix_filter_matches
    echo "nofilter-rc=$?"
    export E2E_MATRIX_FILTER="no_such_suite"
    check_matrix_filter_matches
    echo "unknown-rc=$?"
    export E2E_MATRIX_FILTER="azure_openai_tool_calling"
    printf '%s\\n' "azure_openai_tool_calling" >> '{tmp_path}/.matrix_filter_matched'
    check_matrix_filter_matches
    echo "matched-rc=$?"
    """
    env = {**os.environ, "ARTIFACT_DIR": str(tmp_path)}
    result = subprocess.run(  # noqa: S603
        ["bash", "-c", script],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "nofilter-rc=0" in result.stdout
    assert "unknown-rc=1" in result.stdout
    assert "matched no matrix entry in any suite" in result.stderr
    assert "matched-rc=0" in result.stdout
    assert "E2E_MATRIX_FILTER matched: azure_openai_tool_calling" in (result.stdout)


@pytest.mark.parametrize(
    ("anthropic_key", "expected"),
    [
        ("/keys/anthropic", ["anthropic"]),
        (None, []),
    ],
    ids=["key-present", "key-missing"],
)
def test_cluster_script_gates_anthropic_suite_on_credentials(
    tmp_path: Path, anthropic_key: str | None, expected: list[str]
) -> None:
    """Skip the anthropic suite when its credential is not provided."""
    calls = tmp_path / "calls"
    script = f"""
    source '{SCRIPTS / "utils.sh"}'
    run_suite() {{ printf '%s\\n' "$1" >> '{calls}'; }}
    ANTHROPIC_PROVIDER_KEY_PATH="{anthropic_key or ""}"
    run_suite google_vertex_anthropic "tags" "google_vertex_anthropic" "/keys/vertex" \\
      "claude-haiku-5-5" image default
    if [ -n "${{ANTHROPIC_PROVIDER_KEY_PATH:-}}" ]; then
      run_suite anthropic "tags" "anthropic" "$ANTHROPIC_PROVIDER_KEY_PATH" \\
        "claude-haiku-5-5" image default
    else
      echo "Skipping anthropic suite: ANTHROPIC_PROVIDER_KEY_PATH is not set"
    fi
    """
    env = {**os.environ, "ARTIFACT_DIR": str(tmp_path)}
    subprocess.run(["/bin/bash", "-c", script], env=env, check=True)  # noqa: S603

    suites = [
        line
        for line in calls.read_text().splitlines()
        if "anthropic" in line or "google" in line
    ]
    assert suites == ["google_vertex_anthropic", *expected]


def _run_cluster_script_body(
    script_name: str, tmp_path: Path
) -> subprocess.CompletedProcess:
    """Execute run_suites from a cluster script with make stubbed out."""
    script = (REPO_ROOT / "tests" / "scripts" / script_name).read_text(encoding="utf-8")
    body = script.split("function run_suites() {", 1)[1].split(
        "\nfunction finish()", 1
    )[0]
    calls = tmp_path / "calls"
    shell = (
        f"source '{SCRIPTS / 'utils.sh'}'\n"
        f"make() {{ printf '%s\\n' \"$SUITE_ID\" >> '{calls}'; return 0; }}\n"
        "cleanup_ols_operator() { :; }\n"
        f"function run_suites() {{{body}\n"
        "run_suites\n"
        'echo "suites-rc=$?"\n'
    )
    env = {
        **os.environ,
        "ARTIFACT_DIR": str(tmp_path),
        "DISCONNECTED": "",
        "OLS_IMAGE": "image",
    }
    env.update(
        dict.fromkeys(
            (
                "AZUREOPENAI_PROVIDER_KEY_PATH",
                "OPENAI_PROVIDER_KEY_PATH",
                "VERTEX_PROVIDER_KEY_PATH",
                "WATSONX_PROVIDER_KEY_PATH",
            ),
            "unused",
        )
    )
    return subprocess.run(  # noqa: S603
        ["bash", "-c", shell],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cluster_script_matrix_filter_narrows_only_matrices(
    tmp_path: Path,
) -> None:
    """Run one matrix entry, skip the other matrix, keep other suites.

    A filter naming one matrix's entry must succeed across both matrices.
    """
    calls = tmp_path / "calls"
    os.environ["E2E_MATRIX_FILTER"] = "azure_openai_tool_calling"
    try:
        result = _run_cluster_script_body("test-e2e-cluster.sh", tmp_path)
    finally:
        os.environ.pop("E2E_MATRIX_FILTER", None)

    assert "suites-rc=0" in result.stdout
    lines = calls.read_text().splitlines()
    assert "azure_openai_tool_calling" in lines
    assert "openai_tool_calling" not in lines
    assert "bedrock_openai_tool_calling" not in lines
    # the gpt6 matrix has no matching entry and must skip, not fail
    assert "openai_gpt6_tool_calling" not in lines
    assert "azure_openai_gpt6_tool_calling" not in lines
    # the filter narrows only the tool-calling matrices
    assert "azure_openai" in lines
    assert "watsonx" in lines
    assert "data_export" in lines


def test_cluster_script_matrix_filter_rejects_unknown_id(tmp_path: Path) -> None:
    """A filter matching no entry in any matrix fails the script."""
    os.environ["E2E_MATRIX_FILTER"] = "no_such_suite_anywhere"
    try:
        result = _run_cluster_script_body("test-e2e-cluster.sh", tmp_path)
    finally:
        os.environ.pop("E2E_MATRIX_FILTER", None)

    # run_suites re-enables set -e before returning, so a failing run makes
    # the shell exit before the trailing echo
    assert result.returncode == 1
    assert "matched no matrix entry in any suite" in result.stderr
