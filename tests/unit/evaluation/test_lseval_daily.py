"""Checks for the shared short-dataset CI matrix and daily GCS history."""

import csv
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "tests" / "scripts"


def test_short_suite_matrix_is_shared_and_failure_is_reported(tmp_path):
    """Daily runs all six providers and reports any suite failure."""
    calls = tmp_path / "calls"
    script = f"""
    source '{SCRIPTS / "lseval-short-common.sh"}'
    run_suite() {{ printf '%s|%s|%s\\n' "$1" "$3" "$5" >> '{calls}'; [[ "$3" != watsonx ]]; }}
    cleanup_ols_operator() {{ :; }}
    run_short_lseval_suites daily
    """
    env = {
        **os.environ,
        "OPENAI_PROVIDER_KEY_PATH": "openai",
        "WATSONX_PROVIDER_KEY_PATH": "watsonx",
        "AZUREOPENAI_PROVIDER_KEY_PATH": "azure",
        "VERTEX_PROVIDER_KEY_PATH": "vertex",
        "OLS_IMAGE": "image",
    }
    result = subprocess.run(  # noqa: S603
        ["/bin/bash", "-c", script], env=env, check=False
    )
    assert result.returncode != 0
    assert calls.read_text().splitlines() == [
        "lseval_daily_openai|openai|gpt-5.4-mini",
        "lseval_daily_watsonx|watsonx|ibm/granite-4-h-small",
        "lseval_daily_azure_openai|azure_openai|gpt-5.4-mini",
        "lseval_daily_vertex_gemini|vertex_gemini|gemini-3.1-flash-lite",
        "lseval_daily_vertex_claude|vertex_claude|claude-opus-4-6",
        "lseval_daily_bedrock_deepseek|bedrock_deepseek|deepseek.v3.2",
    ]


def test_presubmit_uses_same_matrix(tmp_path):
    """Presubmit keeps the same six provider/model pairs."""
    calls = tmp_path / "calls"
    script = f"""
    source '{SCRIPTS / "lseval-short-common.sh"}'
    run_suite() {{ printf '%s|%s|%s\\n' "$1" "$3" "$5" >> '{calls}'; }}
    cleanup_ols_operator() {{ :; }}
    run_short_lseval_suites presubmit
    """
    env = {
        **os.environ,
        "OPENAI_PROVIDER_KEY_PATH": "openai",
        "WATSONX_PROVIDER_KEY_PATH": "watsonx",
        "AZUREOPENAI_PROVIDER_KEY_PATH": "azure",
        "VERTEX_PROVIDER_KEY_PATH": "vertex",
        "OLS_IMAGE": "image",
    }
    subprocess.run(["/bin/bash", "-c", script], env=env, check=True)  # noqa: S603
    assert [line.split("|")[0] for line in calls.read_text().splitlines()] == [
        f"lseval_presubmit_{provider}"
        for provider in (
            "openai",
            "watsonx",
            "azure_openai",
            "vertex_gemini",
            "vertex_claude",
            "bedrock_deepseek",
        )
    ]


def test_daily_suite_uses_short_dataset_make_target(tmp_path):
    """The daily suite ID must dispatch to the presubmit pytest entrypoint."""
    calls = tmp_path / "make-targets"
    script = f"""
    source '{SCRIPTS / "utils.sh"}'
    make() {{ printf '%s\\n' "$1" >> '{calls}'; }}
    run_suite lseval_daily_openai lseval openai key gpt-5.4-mini image lseval
    """
    env = {**os.environ, "ARTIFACT_DIR": str(tmp_path)}
    subprocess.run(["/bin/bash", "-c", script], env=env, check=True)  # noqa: S603
    assert calls.read_text().splitlines() == ["test-lseval-presubmit"]


def test_daily_trends_invoked_as_module():
    """The CI entrypoint must use package imports from the repository root."""
    entrypoint = (SCRIPTS / "test-lseval-presubmit.sh").read_text()
    assert "python -m eval.scripts.build_daily_eval_trends" in entrypoint


def test_history_from_latest_prior_daily_artifact(tmp_path):
    """A missing recent snapshot falls back to the next earlier run."""
    from eval.scripts import build_daily_eval_trends as daily

    old = (
        "date,suite,metric,total,pass_rate,error_rate,score_mean\n"
        "2026-01-01,lseval_daily_openai,m,10,90,10,0.9\n"
    )
    names = ["logs/job/100/", "logs/job/102/", "logs/job/101/"]

    def fake_json(url):
        assert "prefix=logs%2Fjob%2F" in url
        return {"prefixes": names}

    def fake_download(url):
        if "102" in url:
            raise daily.HTTPError(url, 404, "missing", {}, None)
        assert "101" in url
        return old.encode()

    with (
        patch.object(daily, "_get_json", side_effect=fake_json),
        patch.object(daily, "_get_bytes", side_effect=fake_download),
    ):
        daily.restore_history(tmp_path / "daily_score_history.csv", "job", "103")
    assert (tmp_path / "daily_score_history.csv").read_text() == old


def test_history_rejects_listing_failure(tmp_path):
    """Do not silently reset score history when GCS is down."""
    from eval.scripts import build_daily_eval_trends as daily

    with patch.object(daily, "_get_json", side_effect=OSError("GCS unavailable")):
        with pytest.raises(OSError, match="GCS unavailable"):
            daily.restore_history(tmp_path / "history.csv", "job", "123")


def test_daily_trends_use_distinct_provider_suites(tmp_path):
    """Write one labeled series per provider and both plot files."""
    from eval.scripts import build_daily_eval_trends as daily

    output = tmp_path / "artifacts"
    for provider in daily.PROVIDERS:
        location = output / "lseval" / provider
        location.mkdir(parents=True)
        (location / "evaluation_20260101_summary.json").write_text(
            json.dumps(
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "summary_stats": {
                        "overall": {"TOTAL": 10, "pass_rate": 90, "error_rate": 10},
                        "by_metric": {"metric": {"score_statistics": {"mean": 0.9}}},
                    },
                }
            )
        )
    history = output / "daily_score_history.csv"
    daily.append_current_summaries(history, output)
    with history.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {row["suite"] for row in rows} == {
        f"lseval_daily_{p}" for p in daily.PROVIDERS
    }
    assert all(row["total"] == "10" for row in rows)
    assert (output / "trend_pass_rate.png").is_file()
    assert (output / "trend_score_mean.png").is_file()


def test_ci_cannot_silently_drop_history_when_build_id_missing(tmp_path):
    """An unidentifiable CI run cannot discard earlier scores."""
    from eval.scripts import build_daily_eval_trends as daily

    with (
        patch.dict(os.environ, {"JOB_NAME": daily.JOB}, clear=True),
        patch(
            "sys.argv", ["build_daily_eval_trends.py", "--artifact-dir", str(tmp_path)]
        ),
    ):
        with pytest.raises(ValueError, match="BUILD_ID"):
            daily.main()


def test_history_rejects_corrupt_snapshot(tmp_path):
    """Corrupt previous snapshots must not be reused."""
    from eval.scripts import build_daily_eval_trends as daily

    with (
        patch.object(daily, "_get_json", return_value={"prefixes": ["logs/job/10/"]}),
        patch.object(daily, "_get_bytes", return_value=b"not,a,daily,history\n"),
    ):
        with pytest.raises(ValueError, match="Invalid daily score history"):
            daily.restore_history(tmp_path / "history.csv", "job", "11")


def test_history_paginates_gcs_listing(tmp_path):
    """Find earlier runs across paginated GCS results."""
    from eval.scripts import build_daily_eval_trends as daily

    old = (
        "date,suite,metric,total,pass_rate,error_rate,score_mean\n"
        "2026-01-01,lseval_daily_openai,m,10,90,10,0.9\n"
    )
    with (
        patch.object(
            daily,
            "_get_json",
            side_effect=[
                {"prefixes": ["logs/job/100/"], "nextPageToken": "next"},
                {"prefixes": ["logs/job/101/"]},
            ],
        ) as listing,
        patch.object(daily, "_get_bytes", return_value=old.encode()) as download,
    ):
        daily.restore_history(tmp_path / "history.csv", "job", "102")
    assert listing.call_count == 2
    assert "pageToken=next" in listing.call_args.args[0]
    assert "101" in download.call_args.args[0]
