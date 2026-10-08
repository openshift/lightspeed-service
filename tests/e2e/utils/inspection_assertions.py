"""Safe assertions for rejected e2e tool results."""

from collections.abc import Mapping

import pytest


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
