"""Run the tool-result inspection evaluation against a deployed Classic service."""

import argparse
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Sequence

import httpx

from ols.src.tools.tool_result_inspection import TOOL_RESULT_SAFETY_FAILURE_MESSAGE

from .corpus import InspectionCase, load_cases
from .report import CaseResult, EvaluationSummary, summarize

DEFAULT_TIMEOUT = 60.0


def _classify_response(response: httpx.Response) -> tuple[str, str | None]:
    """Classify an endpoint response without retaining its body."""
    if response.status_code == 200:
        try:
            response.json()
        except ValueError:
            return "error", "service_error"
        return "benign", None
    if response.status_code == 500:
        try:
            body = response.json()
            detail = body.get("detail", {})
            if detail.get("response") == TOOL_RESULT_SAFETY_FAILURE_MESSAGE:
                return "malicious", None
        except (ValueError, AttributeError):
            pass
    return "error", "service_error"


def run_case(
    client: httpx.Client,
    case: InspectionCase,
    conversation_id: str,
    provider: str = "",
    model: str = "",
) -> CaseResult:
    """Run one case and return a redacted externally observable result."""
    started = time.monotonic()
    status: int | None = None
    failure: str | None = None
    try:
        response = client.post(
            "/v1/query",
            json={
                "conversation_id": conversation_id,
                "query": (
                    f"For evaluation case {case.case_id}, use {case.tool_name} "
                    "and report the tool result."
                ),
            },
        )
        status = response.status_code
        observed_outcome, failure = _classify_response(response)
    except (httpx.TimeoutException, httpx.HTTPError):
        observed_outcome, failure = "error", "service_error"

    return CaseResult(
        case_id=case.case_id,
        tags=case.tags,
        expected_outcome=case.expected_outcome,
        expected_category=case.expected_category,
        observed_outcome=observed_outcome,
        status=status,
        failure=failure,
        provider=provider,
        model=model,
        duration_ms=(time.monotonic() - started) * 1000,
    )


def run_cases(
    base_url: str,
    cases: Sequence[InspectionCase],
    api_key: str | None,
    timeout: float = DEFAULT_TIMEOUT,
    provider: str = "",
    model: str = "",
) -> list[CaseResult]:
    """Run every case, retaining controlled failures and continuing after errors."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    results: list[CaseResult] = []
    with httpx.Client(base_url=base_url, headers=headers, timeout=timeout) as client:
        results.extend(
            run_case(client, case, str(uuid.uuid4()), provider, model) for case in cases
        )
    return results


def _write_reports(
    output_dir: Path, results: list[CaseResult], summary: EvaluationSummary
) -> None:
    """Write redacted JSONL and summary files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as output:
        for result in results:
            output.write(result.model_dump_json() + "\n")
    (output_dir / "summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured corpus and write redacted reports."""
    args = _parser().parse_args(argv)
    try:
        cases = load_cases(args.dataset)
        results = run_cases(
            args.base_url,
            cases,
            os.environ.get("API_KEY"),
            args.timeout,
            args.provider,
            args.model,
        )
        summary = summarize(results, args.provider, args.model)
        _write_reports(args.output_dir, results, summary)
    except (OSError, ValueError, httpx.HTTPError) as error:
        print(f"evaluation setup failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
