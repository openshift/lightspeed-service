"""Tests for tool-result inspection report aggregation."""

from eval.tool_result_inspection.report import CaseResult, summarize


def _result(
    expected: str, observed: str, provider: str = "p", model: str = "m"
) -> CaseResult:
    """Build a redacted result fixture for one observed outcome."""
    failure = "classifier_error" if observed == "error" else None
    return CaseResult(
        case_id=f"{expected}-{observed}-{provider}-{model}",
        tags=["test"],
        expected_outcome=expected,
        expected_category="none" if expected == "benign" else "unknown",
        observed_outcome=observed,
        status=200 if observed == "benign" else 500,
        failure=failure,
        provider=provider,
        model=model,
        duration_ms=1.0,
    )


def test_summarize_reports_confusion_counts_and_provider_groups() -> None:
    """Aggregate outcomes and group metrics by provider and model."""
    results = [
        _result("benign", "benign"),
        _result("benign", "malicious"),
        _result("malicious", "malicious"),
        _result("malicious", "benign"),
        _result("malicious", "error"),
        CaseResult(
            case_id="service-error",
            tags=["test"],
            expected_outcome="benign",
            expected_category="none",
            observed_outcome="error",
            status=None,
            failure="service_error",
            provider="p",
            model="m",
            duration_ms=1.0,
        ),
        _result("benign", "benign", provider="other", model="small"),
    ]

    summary = summarize(results, "p", "m")

    assert summary.total == 7
    assert summary.true_positives == 1
    assert summary.false_positives == 1
    assert summary.false_negatives == 1
    assert summary.classifier_failures == 1
    assert summary.service_errors == 1
    assert summary.groups["p/m"].total == 6
    assert summary.groups["other/small"].true_negatives == 1
    assert summary.groups["p/m"].false_positive_rate == 1 / 2


def test_summarize_handles_empty_and_zero_denominator_groups() -> None:
    """Return zero-safe metrics when a class is absent."""
    summary = summarize([_result("benign", "benign")], "p", "m")

    assert summary.false_positive_rate == 0.0
    assert summary.false_negative_rate == 0.0
    assert summary.groups["p/m"].false_negative_rate == 0.0
    assert summarize([], "p", "m").total == 0
