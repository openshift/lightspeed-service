"""Tests for the codecov coverage upload script (scripts/codecov.sh).

The script uploads coverage reports to codecov.io from prow CI. It must
soft-fail (never block CI), but clearly log whether the upload succeeded,
failed, or was skipped. These tests run the script against a stubbed
``codecovcli`` binary so no network access is required.
"""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CODECOV_SCRIPT = REPO_ROOT / "scripts" / "codecov.sh"

CODECOV_CLI_STUB = """#!/usr/bin/env bash
count="$(cat "$STUB_CALL_COUNT_FILE" 2> /dev/null || echo 0)"
count=$((count + 1))
printf '%s\\n' "$count" > "$STUB_CALL_COUNT_FILE"
printf '%s\\n' "$@" > "$STUB_ARGS_FILE"
if [ "${STUB_FAIL_FIRST_CALL:-0}" = "1" ] && [ "$count" -eq 1 ]; then
    exit 1
fi
exit "${STUB_EXIT_CODE:-0}"
"""

PR_SHA = "a" * 40
BASE_SHA = "b" * 40
TOKEN = "test-codecov-token"  # noqa: S105  # fake token for stubbed CLI


def _write_report_file(tmp_path: Path) -> Path:
    """Create a placeholder coverage report file to upload."""
    report_file = tmp_path / "coverage_unit.xml"
    report_file.write_text("<coverage/>")
    return report_file


def _run_codecov_script(
    tmp_path: Path,
    job_type: str,
    report_file: Path,
    flag: str = "unit",
    token: str | None = TOKEN,
    stub_exit_code: str = "0",
    stub_fail_first_call: bool = False,
    max_attempts: str | None = "1",
) -> subprocess.CompletedProcess[str]:
    """Run scripts/codecov.sh with a stubbed codecovcli on PATH.

    Args:
        tmp_path: pytest temporary directory used for stubs and artifacts.
        job_type: value of the JOB_TYPE environment variable.
        report_file: coverage report file passed as the first argument.
        flag: coverage flag passed as the second argument.
        token: CODECOV_TOKEN value, or None to simulate a missing token.
        stub_exit_code: exit code returned by the stubbed codecovcli.
        stub_fail_first_call: make the stubbed codecovcli fail on its first call.
        max_attempts: CODECOV_MAX_ATTEMPTS value, or None for the script default.

    Returns:
        The completed process with captured stdout and stderr.
    """
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "codecovcli"
    stub.write_text(CODECOV_CLI_STUB)
    stub.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{stub_dir}:{os.environ['PATH']}",
        "JOB_TYPE": job_type,
        "CODECOV_TOKEN": token if token is not None else "",
        "PULL_NUMBER": "1234",
        "PULL_PULL_SHA": PR_SHA,
        "PULL_BASE_SHA": BASE_SHA,
        "PULL_BASE_REF": "main",
        "REPO_OWNER": "openshift",
        "REPO_NAME": "lightspeed-service",
        "ARTIFACT_DIR": str(artifacts_dir),
        "STUB_ARGS_FILE": str(tmp_path / "stub_args.txt"),
        "STUB_CALL_COUNT_FILE": str(tmp_path / "stub_calls.txt"),
        "STUB_EXIT_CODE": stub_exit_code,
        "STUB_FAIL_FIRST_CALL": "1" if stub_fail_first_call else "0",
        "CODECOV_RETRY_SLEEP_SECONDS": "0",
    }
    if max_attempts is not None:
        env["CODECOV_MAX_ATTEMPTS"] = max_attempts
    return subprocess.run(  # noqa: S603
        [str(CODECOV_SCRIPT), str(report_file), flag],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _stub_arguments(tmp_path: Path) -> list[str]:
    """Return the arguments the stubbed codecovcli was invoked with."""
    args_file = tmp_path / "stub_args.txt"
    return args_file.read_text().splitlines()


def _stub_call_count(tmp_path: Path) -> int:
    """Return how many times the stubbed codecovcli was invoked."""
    calls_file = tmp_path / "stub_calls.txt"
    return int(calls_file.read_text().strip())


def _option_value(args: list[str], option: str) -> str:
    """Return the value following the given CLI option in the stub arguments."""
    return args[args.index(option) + 1]


def test_presubmit_upload_passes_reference_flags_as_separate_arguments(tmp_path):
    """PR number, commit SHA and branch must arrive as separate arguments.

    Regression test: quoting the reference flags as a single word previously
    made codecov associate reports with garbage branches like
    " main -C <sha>" (see OLS-4180).
    """
    result = _run_codecov_script(tmp_path, "presubmit", _write_report_file(tmp_path))

    assert result.returncode == 0
    args = _stub_arguments(tmp_path)
    assert _option_value(args, "--pr") == "1234"
    assert _option_value(args, "--sha") == PR_SHA
    assert _option_value(args, "--branch") == "main"


def test_presubmit_upload_passes_common_arguments(tmp_path):
    """The upload carries repository, token, report file and flag metadata."""
    report_file = _write_report_file(tmp_path)
    result = _run_codecov_script(tmp_path, "presubmit", report_file)

    assert result.returncode == 0
    args = _stub_arguments(tmp_path)
    assert "upload-process" in args
    assert _option_value(args, "--git-service") == "github"
    assert _option_value(args, "--slug") == "openshift/lightspeed-service"
    assert _option_value(args, "--token") == TOKEN
    assert _option_value(args, "--file") == str(report_file)
    assert _option_value(args, "--flag") == "unit"
    assert "--fail-on-error" in args
    assert "upload succeeded" in result.stdout


def test_postsubmit_upload_uses_base_ref_and_base_sha(tmp_path):
    """Postsubmit uploads use the base SHA and carry no pull request number."""
    result = _run_codecov_script(tmp_path, "postsubmit", _write_report_file(tmp_path))

    assert result.returncode == 0
    args = _stub_arguments(tmp_path)
    assert _option_value(args, "--sha") == BASE_SHA
    assert _option_value(args, "--branch") == "main"
    assert "--pr" not in args


def test_upload_failure_does_not_block_ci(tmp_path):
    """A failing codecov upload must exit 0 and log the failure loudly."""
    result = _run_codecov_script(
        tmp_path,
        "presubmit",
        _write_report_file(tmp_path),
        stub_exit_code="1",
    )

    assert result.returncode == 0
    assert "CODECOV UPLOAD FAILED" in result.stdout
    assert "not blocked" in result.stdout


def test_transient_failure_is_retried_and_succeeds(tmp_path):
    """A transient codecov failure is retried using the script default attempts.

    CODECOV_MAX_ATTEMPTS is not set, so the script default (2 attempts)
    applies; the stubbed codecovcli fails on its first call only.
    """
    result = _run_codecov_script(
        tmp_path,
        "presubmit",
        _write_report_file(tmp_path),
        stub_fail_first_call=True,
        max_attempts=None,
    )

    assert result.returncode == 0
    assert _stub_call_count(tmp_path) == 2
    assert "upload succeeded" in result.stdout


def test_local_run_skips_upload(tmp_path):
    """Local runs must not upload and must state so in the logs."""
    result = _run_codecov_script(tmp_path, "local", _write_report_file(tmp_path))

    assert result.returncode == 0
    assert not (tmp_path / "stub_args.txt").exists()
    assert "skipped" in result.stdout


def test_missing_token_skips_upload(tmp_path):
    """A missing CODECOV_TOKEN must be reported without blocking CI."""
    result = _run_codecov_script(
        tmp_path, "presubmit", _write_report_file(tmp_path), token=None
    )

    assert result.returncode == 0
    assert not (tmp_path / "stub_args.txt").exists()
    assert "CODECOV UPLOAD FAILED" in result.stdout
    assert "CODECOV_TOKEN" in result.stdout


def test_missing_report_file_does_not_block_ci(tmp_path):
    """A missing coverage report must be reported without blocking CI."""
    result = _run_codecov_script(tmp_path, "presubmit", tmp_path / "nonexistent.xml")

    assert result.returncode == 0
    assert not (tmp_path / "stub_args.txt").exists()
    assert "CODECOV UPLOAD FAILED" in result.stdout
