"""Check that inspection leak failures disclose only the matching source name."""

import pytest

from tests.e2e.utils.inspection_assertions import (
    assert_inspection_outcome,
    assert_rejected_result_absent,
)


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [(["benign"], "benign"), (["malicious"], "malicious")],
)
def test_inspection_outcome_accepts_classifier_decision(
    outcomes: list[str], expected: str
) -> None:
    """Recognize an actual decision rather than an API safety error alone."""
    assert_inspection_outcome(outcomes, expected)


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ([], "malicious"),
        (["classifier_error"], "malicious"),
        (["classifier_error", "malicious"], "malicious"),
        (["malicious"], "benign"),
    ],
)
def test_inspection_outcome_rejects_missing_or_failed_decision(
    outcomes: list[str], expected: str
) -> None:
    """Fail when no classifier decision was made or classification failed."""
    with pytest.raises(pytest.fail.Exception):
        assert_inspection_outcome(outcomes, expected)


def test_inspection_leak_report_names_source_without_content() -> None:
    """A failing check reports the source, not the rejected tool result."""
    with pytest.raises(pytest.fail.Exception) as exc_info:
        assert_rejected_result_absent(
            {"response_text": False, "service_logs": True, "mock_logs": False}
        )
    message = str(exc_info.value)
    assert "service_logs" in message
    assert "response_text" not in message
    assert "mock_logs" not in message


def test_inspection_leak_report_accepts_clean_sources() -> None:
    """The leak check accepts responses and logs without rejected output."""
    assert_rejected_result_absent({"response_text": False, "service_logs": False})
