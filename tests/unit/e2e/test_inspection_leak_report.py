"""Check that inspection leak failures disclose only the matching source name."""

import pytest

from tests.e2e.utils import inspection_assertions
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


def test_classifier_failure_report_extracts_only_safe_metadata() -> None:
    """Extract temporary classifier diagnostics without exception messages."""
    extract = getattr(inspection_assertions, "classifier_failures_since", None)
    assert callable(extract)
    previous = "older service logs\n"
    current = (
        previous
        + '{"message":"TEMPORARY tool_result_classifier_failure '
        + "stage=invoke attempt=1/3 error_type=APIStatusError "
        + 'cause_types=HTTPStatusError status_code=503"}\n'
        + "provider exception: api_key=must-not-appear\n"
    )

    failures = extract(previous, current)

    assert failures == ["invoke/1:APIStatusError:causes=HTTPStatusError:status=503"]
    assert "must-not-appear" not in str(failures)


def test_failure_diagnostics_report_only_safe_request_metadata() -> None:
    """Failure details show status and outcomes, not response or tool content."""
    formatter = getattr(
        inspection_assertions, "format_inspection_failure_diagnostics", None
    )
    assert callable(formatter)
    assert formatter(502, True, [("classifier_error", "result")]) == (
        "HTTP status=502; mock_tool_called=true; "
        "inspection_outcomes=classifier_error/result; classifier_failures=unavailable"
    )
    assert formatter(500, None, None) == (
        "HTTP status=500; mock_tool_called=unavailable; "
        "inspection_outcomes=unavailable; classifier_failures=unavailable"
    )
    assert formatter(
        None,
        True,
        [("classifier_error", "result")],
        ["invoke/1:APIStatusError:causes=HTTPStatusError:status=503", "secret=hidden"],
    ) == (
        "HTTP status=unavailable; mock_tool_called=true; "
        "inspection_outcomes=classifier_error/result; "
        "classifier_failures=invoke/1:APIStatusError:causes=HTTPStatusError:status=503"
    )
