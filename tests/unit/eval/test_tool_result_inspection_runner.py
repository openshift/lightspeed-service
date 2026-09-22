"""Tests for the tool-result inspection evaluation runner."""

import json

import httpx

from eval.tool_result_inspection import runner
from eval.tool_result_inspection.corpus import InspectionCase
from eval.tool_result_inspection.report import CaseResult
from eval.tool_result_inspection.runner import run_case

FIXED_MESSAGE = "Lightspeed stopped the operation because a tool result failed the safety inspection."  # noqa: E501


def _case(case_id: str = "case-1") -> InspectionCase:
    """Build a malicious inspection case fixture."""
    return InspectionCase(
        case_id=case_id,
        tool_name="get_pods",
        result_type="result",
        content="private tool output must not be recorded",
        expected_outcome="malicious",
        expected_category="instruction_override",
        tags=["attack"],
    )


def _client(handler) -> httpx.Client:
    """Build an HTTP client backed by the supplied mock transport."""
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://test",
        headers={"Authorization": "Bearer test-key"},
    )


def test_run_case_classifies_normal_success_without_recording_content() -> None:
    """Classify a successful service response as benign."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a successful redacted service response."""
        assert request.url.path == "/v1/query"
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"response": "safe answer"})

    result = run_case(
        _client(handler), _case("benign"), "conversation-1", "provider", "model"
    )

    assert result.observed_outcome == "benign"
    assert result.status == 200
    assert result.failure is None
    assert "private tool output" not in result.model_dump_json()


def test_run_case_classifies_exact_safety_failure() -> None:
    """Classify only the fixed safety message as a malicious stop."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the fixed safety-stop response."""
        return httpx.Response(
            500,
            json={"detail": {"response": FIXED_MESSAGE, "cause": ""}},
        )

    result = run_case(_client(handler), _case(), "conversation-1", "provider", "model")

    assert result.observed_outcome == "malicious"
    assert result.status == 500
    assert result.failure is None


def test_run_case_records_controlled_failures_without_response_body() -> None:
    """Record unexpected, malformed, timeout, and connection failures safely."""
    responses = [
        httpx.Response(500, json={"detail": {"response": "secret body"}}),
        httpx.Response(200, content=b"not-json"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        """Return controlled failures without exposing response content."""
        if responses:
            return responses.pop(0)
        raise httpx.ConnectError("connection secret")

    first = run_case(_client(handler), _case("unexpected"), "c1", "p", "m")
    second = run_case(_client(handler), _case("malformed"), "c2", "p", "m")
    third = run_case(_client(handler), _case("connection"), "c3", "p", "m")

    assert first.observed_outcome == "error"
    assert first.failure == "service_error"
    assert second.failure == "service_error"
    assert third.failure == "service_error"
    assert all(
        forbidden not in result.model_dump_json()
        for result in (first, second, third)
        for forbidden in ("secret body", "connection secret")
    )


def test_main_writes_redacted_reports(tmp_path, monkeypatch) -> None:
    """Write valid report files without serializing case content."""
    dataset = tmp_path / "cases.yaml"
    dataset.write_text(
        "- case_id: cli-case\n"
        "  tool_name: get_pods\n"
        "  result_type: result\n"
        "  content: committed-secret-content\n"
        "  expected_outcome: benign\n"
        "  expected_category: none\n"
        "  tags: [benign]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "run_cases",
        lambda *args: [
            CaseResult(
                case_id="cli-case",
                tags=["benign"],
                expected_outcome="benign",
                expected_category="none",
                observed_outcome="benign",
                status=200,
                failure=None,
                provider="p",
                model="m",
                duration_ms=1,
            )
        ],
    )

    assert (
        runner.main(
            [
                "--base-url",
                "http://test",
                "--dataset",
                str(dataset),
                "--output-dir",
                str(tmp_path / "results"),
                "--provider",
                "p",
                "--model",
                "m",
            ]
        )
        == 0
    )
    results = (tmp_path / "results" / "results.jsonl").read_text()
    summary = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert "committed-secret-content" not in results
    assert "committed-secret-content" not in json.dumps(summary)
