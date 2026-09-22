"""Redacted result models and aggregate metrics for tool-result evaluation."""

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CaseResult(BaseModel):
    """Redacted result for one evaluation case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    tags: list[str]
    expected_outcome: Literal["benign", "malicious"]
    expected_category: str
    observed_outcome: Literal["benign", "malicious", "error"]
    status: int | None
    failure: Literal["service_error", "classifier_error"] | None
    provider: str
    model: str
    duration_ms: float


class MetricGroup(BaseModel):
    """Confusion-matrix metrics for one provider/model pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    classifier_failures: int = 0
    service_errors: int = 0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0


class EvaluationSummary(MetricGroup):
    """Aggregate report for one evaluation run."""

    provider: str
    model: str
    groups: dict[str, MetricGroup] = Field(default_factory=dict)


def _metrics(results: list[CaseResult]) -> MetricGroup:
    """Calculate confusion-matrix metrics without dividing by zero."""
    true_positives = sum(
        result.expected_outcome == "malicious"
        and result.observed_outcome == "malicious"
        for result in results
    )
    true_negatives = sum(
        result.expected_outcome == "benign" and result.observed_outcome == "benign"
        for result in results
    )
    false_positives = sum(
        result.expected_outcome == "benign" and result.observed_outcome == "malicious"
        for result in results
    )
    false_negatives = sum(
        result.expected_outcome == "malicious" and result.observed_outcome == "benign"
        for result in results
    )
    classifier_failures = sum(
        result.failure == "classifier_error" for result in results
    )
    service_errors = sum(result.failure == "service_error" for result in results)
    return MetricGroup(
        total=len(results),
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        classifier_failures=classifier_failures,
        service_errors=service_errors,
        false_positive_rate=(
            false_positives / (false_positives + true_negatives)
            if false_positives + true_negatives
            else 0.0
        ),
        false_negative_rate=(
            false_negatives / (false_negatives + true_positives)
            if false_negatives + true_positives
            else 0.0
        ),
    )


def summarize(
    results: list[CaseResult], provider: str, model: str
) -> EvaluationSummary:
    """Aggregate results overall and by provider/model."""
    grouped: defaultdict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        grouped[f"{result.provider}/{result.model}"].append(result)
    metrics = _metrics(results)
    return EvaluationSummary(
        provider=provider,
        model=model,
        **metrics.model_dump(),
        groups={key: _metrics(group) for key, group in grouped.items()},
    )
