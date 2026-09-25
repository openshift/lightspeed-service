"""Publish per-provider daily score history and plots from immutable Prow artifacts.

The previous successful snapshot is recovered from the public Prow GCS bucket;
the updated CSV and plots are uploaded automatically with the current job's
ARTIFACT_DIR. Requires the bucket to allow anonymous object listing and reads.
"""

import argparse
import csv
import io
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import urlopen

from eval.scripts import update_eval_trends as trends

BUCKET = "test-platform-results-public"
JOB = "periodic-ci-openshift-lightspeed-service-main-4.22-ols-eval-periodic-daily"
TARGET = "ols-eval-periodic-daily"
PROVIDERS = (
    "openai",
    "watsonx",
    "azure_openai",
    "vertex_gemini",
    "vertex_claude",
    "bedrock_deepseek",
)
COLUMNS = ("date", "suite", "metric", "total", "pass_rate", "error_rate", "score_mean")
BASE_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"


def _get_json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:  # noqa: S310 - fixed GCS host
        return json.load(response)


def _get_bytes(url: str) -> bytes:
    with urlopen(url, timeout=30) as response:  # noqa: S310 - fixed GCS host
        return response.read()


def restore_history(history: Path, job: str, build_id: str) -> None:
    """Find the newest earlier job with a published daily history snapshot."""
    prefix = f"logs/{job}/"
    page_token = None
    builds: list[str] = []
    while True:
        query = {"prefix": prefix, "delimiter": "/", "fields": "nextPageToken,prefixes"}
        if page_token:
            query["pageToken"] = page_token
        listing = _get_json(f"{BASE_URL}?{urlencode(query)}")
        builds.extend(
            folder.removeprefix(prefix).rstrip("/")
            for folder in listing.get("prefixes", [])
            if folder.startswith(prefix)
        )
        page_token = listing.get("nextPageToken")
        if not page_token:
            break

    for previous in sorted(
        (b for b in builds if b.isdigit() and int(b) < int(build_id)),
        key=int,
        reverse=True,
    ):
        object_name = f"{prefix}{previous}/artifacts/{TARGET}/e2e/artifacts/daily_score_history.csv"
        url = f"{BASE_URL}/{quote(object_name, safe='')}?alt=media"
        try:
            contents = _get_bytes(url).decode("utf-8")
        except HTTPError as exc:
            if exc.code == 404:
                continue  # Failed/incomplete job: try an older snapshot.
            raise
        rows = csv.DictReader(io.StringIO(contents))
        if tuple(rows.fieldnames or ()) != COLUMNS or any(
            not row["suite"].startswith("lseval_daily_") for row in rows
        ):
            raise ValueError(f"Invalid daily score history in {object_name}")
        history.write_text(contents, encoding="utf-8")
        print(f"Restored daily history from gs://{BUCKET}/{object_name}")
        return
    print("No earlier daily history snapshot found; starting a new history")


def append_current_summaries(history: Path, artifact_dir: Path) -> None:
    """Add available provider summaries to the daily CSV (never weekly results)."""
    rows = []
    for provider in PROVIDERS:
        summaries = list(
            (artifact_dir / "lseval" / provider).glob("evaluation_*_summary.json")
        )
        if not summaries:
            print(f"No daily summary for {provider}; omitting this provider")
            continue
        summary = json.loads(
            max(summaries, key=lambda p: p.stat().st_mtime).read_text()
        )
        rows.extend(
            trends._rows_from_summary(
                summary, f"lseval_daily_{provider}", trends._run_date(summary, None)
            )
        )
    if rows:
        trends._append_to_history(history, rows)
    if history.exists():
        trends._generate_plots(history, artifact_dir, frequency="Daily")


def main() -> None:
    """Recover previous history, append this run, and publish daily plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--build-id", default=os.getenv("BUILD_ID"))
    parser.add_argument("--job", default=JOB)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    history = args.artifact_dir / "daily_score_history.csv"
    if os.getenv("JOB_NAME") and not args.build_id:
        raise ValueError("BUILD_ID is required in CI to restore daily score history")
    if args.build_id:
        restore_history(history, args.job, args.build_id)
    else:
        print("No BUILD_ID (local run): starting daily history from local artifacts")
    append_current_summaries(history, args.artifact_dir)


if __name__ == "__main__":
    main()
