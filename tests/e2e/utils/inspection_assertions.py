"""Safe assertions for rejected e2e tool results."""

import re
from collections.abc import Mapping

import pytest

_CLASSIFIER_FAILURE_LOG = re.compile(
    r"TEMPORARY tool_result_classifier_failure stage=(invoke|parse|quota) "
    r"attempt=([1-3])/3 error_type=([A-Za-z_][A-Za-z0-9_]*) "
    r"cause_types=(none|[A-Za-z_][A-Za-z0-9_]*(?:,[A-Za-z_][A-Za-z0-9_]*)*) "
    r"status_code=(unknown|[1-5][0-9]{2})(?![0-9])"
)
_SAFE_CLASSIFIER_FAILURE = re.compile(
    r"(?:invoke|parse|quota)/[1-3]:[A-Za-z_][A-Za-z0-9_]*:"
    r"causes=(?:none|[A-Za-z_][A-Za-z0-9_]*(?:,[A-Za-z_][A-Za-z0-9_]*)*):"
    r"status=(?:unknown|[1-5][0-9]{2})"
)


def classifier_failures_since(previous: str, current: str) -> list[str]:
    """Extract only structured temporary classifier failure metadata."""
    recent = current[len(previous) :] if current.startswith(previous) else ""
    failures = []
    for match in _CLASSIFIER_FAILURE_LOG.finditer(recent):
        stage, attempt, error_type, cause_types, status_code = match.groups()
        failures.append(
            f"{stage}/{attempt}:{error_type}:causes={cause_types}:status={status_code}"
        )
    return failures


def format_inspection_failure_diagnostics(
    status_code: int | None,
    mock_tool_called: bool | None,
    outcomes: list[tuple[str, str]] | None,
    classifier_failures: list[str] | None = None,
) -> str:
    """Format allowlisted failure metadata without request or tool content."""
    mock_call = (
        "unavailable" if mock_tool_called is None else str(mock_tool_called).lower()
    )
    if outcomes is None:
        outcome_summary = "unavailable"
    else:
        safe_outcomes = [
            f"{outcome}/{result_type}"
            for outcome, result_type in outcomes
            if outcome in {"benign", "malicious", "classifier_error"}
            and result_type in {"result", "error"}
        ]
        outcome_summary = ",".join(safe_outcomes) or "none"
    if classifier_failures is None:
        failure_summary = "unavailable"
    else:
        safe_failures = [
            failure
            for failure in classifier_failures
            if _SAFE_CLASSIFIER_FAILURE.fullmatch(failure)
        ]
        failure_summary = ",".join(safe_failures) or "none"
    status = status_code if status_code is not None else "unavailable"
    return (
        f"HTTP status={status}; mock_tool_called={mock_call}; "
        f"inspection_outcomes={outcome_summary}; "
        f"classifier_failures={failure_summary}"
    )


def assert_inspection_outcome(outcomes: list[str], expected: str) -> None:
    """Require an actual classifier decision, not a fail-closed provider error."""
    if (
        expected not in outcomes
        or "classifier_error" in outcomes
        or (expected == "benign" and "malicious" in outcomes)
    ):
        pytest.fail(f"Expected inspection outcome {expected}; observed {outcomes}")


def assert_rejected_result_absent(source_matches: Mapping[str, bool]) -> None:
    """Report only source names; never include untrusted content in pytest failures."""
    leaking_sources = [name for name, matched in source_matches.items() if matched]
    if leaking_sources:
        pytest.fail(f"Rejected tool result found in: {', '.join(leaking_sources)}")
