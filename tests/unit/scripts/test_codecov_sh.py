"""Tests for the codecov coverage upload script (scripts/codecov.sh).

The script uploads coverage reports to codecov.io from prow CI. It must
soft-fail (never block CI), but clearly log whether the upload succeeded,
failed, or was skipped. These tests run the script against a stubbed
``codecovcli`` binary so no network access is required.
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CODECOV_SCRIPT = REPO_ROOT / "scripts" / "codecov.sh"

CODECOV_CLI_STUB = """#!/usr/bin/env bash
count="$(cat "$STUB_CALL_COUNT_FILE" 2> /dev/null || echo 0)"
count=$((count + 1))
printf '%s\\n' "$count" > "$STUB_CALL_COUNT_FILE"
printf '%s\\n' "$@" > "$STUB_ARGS_FILE"
if [ -n "${STUB_CHILD_PID_FILE:-}" ]; then
    sleep 30 &
    child_pid=$!
    printf '%s\\n' "$child_pid" > "$STUB_CHILD_PID_FILE"
    wait "$child_pid"
fi
if [ -n "${STUB_SLEEP_SECONDS:-}" ]; then
    sleep "$STUB_SLEEP_SECONDS"
fi
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
    timeout_seconds: str = "300",
    stub_sleep_seconds: str = "",
    child_pid_file: Path | None = None,
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
        timeout_seconds: Deadline for each upload attempt.
        stub_sleep_seconds: Delay used to simulate a stuck CLI.
        child_pid_file: File used to record a spawned child process.

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
        "CODECOV_TIMEOUT_SECONDS": timeout_seconds,
        "STUB_SLEEP_SECONDS": stub_sleep_seconds,
        "STUB_CHILD_PID_FILE": str(child_pid_file) if child_pid_file else "",
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


@pytest.mark.parametrize(
    "cancel_signal", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]
)
def test_perl_helper_cancellation_kills_command_group(
    tmp_path: Path, cancel_signal: signal.Signals
) -> None:
    """Stop the command and its descendants when the Perl helper is cancelled."""
    pid_file = tmp_path / "pids.txt"
    command = (
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "Path(sys.argv[1]).write_text(f'{os.getpid()} {child.pid}')\n"
        "time.sleep(30)\n"
    )
    process = subprocess.Popen(  # noqa: S603
        [  # noqa: S607
            "perl",
            str(REPO_ROOT / "scripts" / "run_with_timeout.pl"),
            "30",
            sys.executable,
            "-c",
            command,
            str(pid_file),
        ],
        start_new_session=True,
    )
    pids: list[int] = []
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if pid_file.exists():
                pids = [int(pid) for pid in pid_file.read_text().split()]
                if len(pids) == 2:
                    break
            time.sleep(0.02)
        assert len(pids) == 2, "command and descendant did not start"

        process.send_signal(cancel_signal)
        assert process.wait(timeout=8) == 128 + cancel_signal
        for pid in pids:
            for _ in range(30):
                status = subprocess.run(  # noqa: S603
                    ["ps", "-o", "stat=", "-p", str(pid)],  # noqa: S607
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
                if not status or status.startswith("Z"):
                    break
                time.sleep(0.1)
            assert not status or status.startswith(
                "Z"
            ), f"process {pid} is still running"
    finally:
        if pids:
            try:
                os.killpg(pids[0], signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_zero_timeout_disables_the_deadline(tmp_path):
    """CODECOV_TIMEOUT_SECONDS=0 uploads instead of failing every attempt."""
    result = _run_codecov_script(
        tmp_path, "presubmit", _write_report_file(tmp_path), timeout_seconds="0"
    )

    assert result.returncode == 0
    assert _stub_call_count(tmp_path) == 1
    assert "upload succeeded" in result.stdout


def _run_timeout_helper(
    *args: str, env: dict[str, str] | None = None
) -> tuple[int, float]:
    """Run scripts/run_with_timeout.pl with the given arguments."""
    started = time.monotonic()
    completed = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "perl",
            str(REPO_ROOT / "scripts" / "run_with_timeout.pl"),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **(env or {})},
    )
    return completed.returncode, time.monotonic() - started


def test_timeout_helper_treats_zero_as_no_deadline() -> None:
    """A zero deadline disables the limit, as GNU timeout's zero does."""
    exit_code, _ = _run_timeout_helper(
        "0", sys.executable, "-c", "import sys, time; time.sleep(1); sys.exit(3)"
    )

    assert exit_code == 3


def test_timeout_helper_accepts_fractional_deadline() -> None:
    """A fractional deadline such as 0.5 expires instead of being rejected."""
    exit_code, elapsed = _run_timeout_helper(
        "0.5", sys.executable, "-c", "import time; time.sleep(30)"
    )

    assert exit_code == 124
    assert elapsed < 5


def test_timeout_helper_keeps_deadline_without_time_hires(tmp_path: Path) -> None:
    """A perl without Time::HiRes still enforces the deadline.

    Time::HiRes is a separate package on RHEL, so the prow image runs a perl
    that lacks it. The helper must fall back to whole-second alarms instead of
    failing at compile time and letting the upload run unbounded.
    """
    shadow = tmp_path / "shadow"
    (shadow / "Time").mkdir(parents=True)
    (shadow / "Time" / "HiRes.pm").write_text('die "Time::HiRes unavailable\n";\n1;\n')

    exit_code, elapsed = _run_timeout_helper(
        "0.5",
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        env={"PERL5LIB": str(shadow)},
    )

    assert exit_code == 124
    assert elapsed < 5


def test_upload_timeout_stops_cli_and_soft_fails(tmp_path):
    """Stop a stuck Codecov CLI at the configured timeout."""
    started = time.monotonic()
    result = _run_codecov_script(
        tmp_path,
        "presubmit",
        _write_report_file(tmp_path),
        timeout_seconds="1",
        stub_sleep_seconds="30",
    )
    elapsed = time.monotonic() - started

    assert result.returncode == 0
    assert _stub_call_count(tmp_path) == 1
    assert elapsed < 8
    assert "CODECOV UPLOAD FAILED" in result.stdout


@pytest.mark.parametrize(
    "force_perl", [False, True], ids=["host-path", "perl-fallback"]
)
def test_upload_timeout_kills_cli_descendants(tmp_path, monkeypatch, force_perl):
    """Stop descendants using the host backend or the explicitly selected Perl fallback."""
    if force_perl:
        bin_dir = tmp_path / "perl-bin"
        bin_dir.mkdir()
        for command in ("bash", "cat", "dirname", "perl", "ps", "sleep"):
            executable = shutil.which(command)
            assert executable is not None, f"required command {command} is unavailable"
            (bin_dir / command).symlink_to(executable)
        monkeypatch.setenv("PATH", str(bin_dir))

    child_pid_file = tmp_path / "child_pid.txt"
    result = _run_codecov_script(
        tmp_path,
        "presubmit",
        _write_report_file(tmp_path),
        timeout_seconds="1",
        child_pid_file=child_pid_file,
    )

    assert result.returncode == 0
    assert _stub_call_count(tmp_path) == 1
    assert "CODECOV UPLOAD FAILED" in result.stdout
    child_pid = int(child_pid_file.read_text().strip())

    for _ in range(30):
        status = subprocess.run(  # noqa: S603
            ["ps", "-o", "stat=", "-p", str(child_pid)],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if not status or status.startswith("Z"):
            break
        time.sleep(0.1)

    assert not status or status.startswith(
        "Z"
    ), f"child process {child_pid} is still running"
