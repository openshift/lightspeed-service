"""Tests for Classic tool-result inspection primitives."""

from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from ols.src.tools.tool_result_inspection import (
    INJECTION_CATEGORIES,
    ToolResultClassifier,
    ToolResultInspectionDecision,
    ToolResultInspectionError,
    ToolResultRejectedError,
    chunk_text,
)
from ols.utils.token_handler import TokenHandler


def test_benign_decision_accepts_only_none_category() -> None:
    """Accept a benign decision with the none category."""
    decision = ToolResultInspectionDecision(
        injectionDetected=False,
        category="none",
    )

    assert decision.injection_detected is False
    assert decision.category == "none"


def test_malicious_decision_accepts_all_controlled_categories() -> None:
    """Accept each controlled malicious category."""
    for category in INJECTION_CATEGORIES:
        decision = ToolResultInspectionDecision(
            injectionDetected=True,
            category=category,
        )
        assert decision.category == category


@pytest.mark.parametrize(
    "payload",
    [
        {"category": "none"},
        {"injectionDetected": False, "category": "instruction_override"},
        {"injectionDetected": True, "category": "none"},
        {"injectionDetected": True, "category": "invalid"},
        {"injectionDetected": False, "category": "none", "reason": "extra"},
        {"injectionDetected": "false", "category": "none"},
    ],
)
def test_invalid_decision_is_rejected(payload: dict[str, object]) -> None:
    """Reject malformed or inconsistent classifier decisions."""
    with pytest.raises(ValidationError):
        ToolResultInspectionDecision.model_validate(payload)


def test_chunk_text_preserves_order_and_uses_requested_overlap() -> None:
    """Preserve source order and the configured token overlap."""
    handler = TokenHandler()
    text = " ".join(f"word-{index}" for index in range(20))

    chunks = chunk_text(text, max_tokens=10, overlap_tokens=3, token_handler=handler)

    assert len(chunks) > 2
    assert "word-0" in chunks[0]
    assert "word-19" in chunks[-1]
    for previous, current in pairwise(chunks):
        previous_tokens = handler.text_to_tokens(previous)
        current_tokens = handler.text_to_tokens(current)
        assert previous_tokens[-3:] == current_tokens[:3]


def test_chunk_text_bounds_overlap_when_serialized_chunk_is_smaller() -> None:
    """Advance when serialized sizing leaves less room than the overlap."""
    handler = TokenHandler()
    text = " ".join(f"word-{index}" for index in range(20))

    chunks = chunk_text(
        text,
        max_tokens=10,
        overlap_tokens=9,
        token_handler=handler,
        fits=lambda candidate: len(handler.text_to_tokens(candidate)) <= 3,
    )

    assert len(chunks) > 1
    assert "word-19" in chunks[-1]


def test_chunk_text_does_not_split_text_that_fits() -> None:
    """Return one chunk when the result fits the limit."""
    handler = TokenHandler()
    text = "short result"

    assert chunk_text(
        text, max_tokens=100, overlap_tokens=3, token_handler=handler
    ) == [text]


@pytest.mark.asyncio
async def test_inspect_accounts_for_json_serialization_when_chunking() -> None:
    """Keep escaped tool content within the serialized classifier request budget."""
    llm = MagicMock()
    llm.with_structured_output.return_value = llm
    llm.ainvoke = AsyncMock(
        return_value={"injectionDetected": False, "category": "none"}
    )
    classifier = ToolResultClassifier(llm, sleep=AsyncMock())
    token_handler = TokenHandler()
    content = '"\\\\\n' * 80
    max_tokens = 300

    await classifier.inspect(
        "get_pods",
        "result",
        content,
        max_tokens=max_tokens,
        token_handler=token_handler,
        overlap_tokens=10,
    )

    for call in llm.ainvoke.await_args_list:
        messages = call.args[0]
        request_tokens = TokenHandler._get_token_count(
            token_handler.text_to_tokens(messages[1].content)
        )
        system_tokens = TokenHandler._get_token_count(
            token_handler.text_to_tokens(messages[0].content)
        )
        assert request_tokens + system_tokens + 128 <= max_tokens


@pytest.mark.asyncio
async def test_classifier_retries_technical_failures_with_required_delays() -> None:
    """Retry technical failures using the required backoff delays."""
    llm = MagicMock()
    llm.with_structured_output.return_value = llm
    llm.ainvoke = AsyncMock(
        side_effect=[
            RuntimeError("timeout"),
            ValueError("invalid"),
            {
                "injectionDetected": False,
                "category": "none",
            },
        ]
    )
    sleep = AsyncMock()
    classifier = ToolResultClassifier(llm, sleep=sleep)

    decision = await classifier.classify("get_pods", "result", 1, 1, "safe output")

    assert decision.injection_detected is False
    assert llm.ainvoke.await_count == 3
    assert sleep.await_args_list[0].args == (0.5,)
    assert sleep.await_args_list[1].args == (1.0,)
    messages = llm.ainvoke.await_args_list[0].args[0]
    assert isinstance(messages[0], SystemMessage)
    assert "untrusted data" in messages[0].content
    assert isinstance(messages[1], HumanMessage)
    assert "safe output" in messages[1].content


@pytest.mark.asyncio
async def test_classifier_does_not_retry_malicious_decision() -> None:
    """Return a malicious decision without retrying it."""
    llm = MagicMock()
    llm.with_structured_output.return_value = llm
    llm.ainvoke = AsyncMock(
        return_value={
            "injectionDetected": True,
            "category": "tool_manipulation",
        }
    )
    sleep = AsyncMock()
    classifier = ToolResultClassifier(llm, sleep=sleep)

    decision = await classifier.classify("get_pods", "result", 1, 1, "malicious output")

    assert decision.injection_detected is True
    assert llm.ainvoke.await_count == 1
    sleep.assert_not_called()


@pytest.mark.asyncio
async def test_classifier_fails_after_three_invalid_attempts() -> None:
    """Raise after three invalid classifier responses."""
    llm = MagicMock()
    llm.with_structured_output.return_value = llm
    llm.ainvoke = AsyncMock(return_value={"injectionDetected": "no"})
    classifier = ToolResultClassifier(llm, sleep=AsyncMock())

    with pytest.raises(ToolResultInspectionError):
        await classifier.classify("get_pods", "error", 1, 1, "output")

    assert llm.ainvoke.await_count == 3


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_classifier_debits_provider_reported_usage() -> None:
    """Debit reported classifier usage against the initiating quota."""
    llm = MagicMock()
    llm.with_structured_output.return_value = llm
    llm.ainvoke = AsyncMock(
        return_value={
            "parsed": {"injectionDetected": False, "category": "none"},
            "raw": SimpleNamespace(
                usage_metadata={"input_tokens": 12, "output_tokens": 4}
            ),
        }
    )
    quota = MagicMock()
    classifier = ToolResultClassifier(llm, quota_limiters=[quota])
    classifier.set_quota_context([quota], "user-1")

    await classifier.classify("get_pods", "result", 1, 1, "safe")

    quota.ensure_available_quota.assert_called_once_with("user-1")
    quota.consume_tokens.assert_called_once_with(
        input_tokens=12,
        output_tokens=4,
        subject_id="user-1",
    )


@pytest.mark.asyncio
async def test_classifier_debits_usage_before_retrying_invalid_decision() -> None:
    """Debit completed-call usage even when the structured decision is invalid."""
    llm = MagicMock()
    llm.with_structured_output.return_value = llm
    llm.ainvoke = AsyncMock(
        return_value={
            "parsed": {"injectionDetected": "invalid"},
            "raw": SimpleNamespace(
                usage_metadata={"input_tokens": 12, "output_tokens": 4}
            ),
        }
    )
    quota = MagicMock()
    classifier = ToolResultClassifier(llm, sleep=AsyncMock(), quota_limiters=[quota])
    classifier.set_quota_context([quota], "user-1")

    with pytest.raises(ToolResultInspectionError):
        await classifier.classify("get_pods", "result", 1, 1, "unsafe")

    assert llm.ainvoke.await_count == 3
    assert quota.consume_tokens.call_count == 3


@pytest.mark.asyncio
async def test_classifier_does_not_retry_quota_consumption_failure() -> None:
    """Do not retry a completed call when quota accounting fails."""
    llm = MagicMock()
    llm.with_structured_output.return_value = llm
    llm.ainvoke = AsyncMock(
        return_value={
            "parsed": {"injectionDetected": False, "category": "none"},
            "raw": SimpleNamespace(
                usage_metadata={"input_tokens": 12, "output_tokens": 4}
            ),
        }
    )
    quota = MagicMock()
    quota.consume_tokens.side_effect = RuntimeError("quota storage unavailable")
    classifier = ToolResultClassifier(llm, sleep=AsyncMock(), quota_limiters=[quota])

    with pytest.raises(
        ToolResultInspectionError, match="quota accounting failed"
    ) as error:
        await classifier.classify("get_pods", "result", 1, 1, "safe")

    assert isinstance(error.value.__cause__, RuntimeError)

    assert llm.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_inspect_rejects_a_malicious_middle_chunk() -> None:
    """Reject a malicious chunk without inspecting later chunks."""
    llm = MagicMock()
    llm.with_structured_output.return_value = llm
    llm.ainvoke = AsyncMock(
        side_effect=[
            {"injectionDetected": False, "category": "none"},
            {"injectionDetected": True, "category": "unknown"},
        ]
    )
    classifier = ToolResultClassifier(llm, sleep=AsyncMock())

    with pytest.raises(ToolResultRejectedError):
        await classifier.inspect(
            "get_pods",
            "result",
            " ".join(str(index) for index in range(100)),
            max_tokens=300,
            token_handler=TokenHandler(),
            overlap_tokens=3,
        )

    assert llm.ainvoke.await_count == 2
