"""Structured tool-output redaction and bounded traversal tests."""

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    computed_field,
    field_serializer,
)

from ols.app.models.models import LLMResponse, RagChunk
from ols.src.redaction.rulesets.mcp import MCP_RULES
from ols.src.redaction.tool_output import redact_tool_output


class _Document(BaseModel):
    """Typed tool metadata with optional extra fields."""

    model_config = ConfigDict(extra="allow")
    text: str


class _ComputedDocument(BaseModel):
    """Expose a computed value backed by private state."""

    _address: str = PrivateAttr(default="admin@example.com")

    @computed_field
    @property
    def address(self) -> str:
        """Return the original private value whenever the model is serialized."""
        return self._address


class _SerializedDocument(BaseModel):
    """Regenerate a value during field serialization."""

    address: str = "admin@example.com"

    @field_serializer("address")
    def serialize_address(self, value: str) -> str:
        """Return source data independently of a copied field value."""
        return "admin@example.com"


def test_metadata_only_matches_are_reported_without_mutation() -> None:
    """Sanitize references, models, mapping keys, and lists without mutating originals."""
    document = RagChunk("admin@example.com", "https://example.com", "Contact")
    model = _Document(text="admin@example.com")
    structured = {"model": model, "extra@example.com": [None, True, 12, 1.5, ("safe",)]}
    text, safe, references, info = redact_tool_output(
        "safe text", structured, [document], MCP_RULES.compile(), 4000
    )
    assert text == "safe text"
    assert safe["model"] == {"text": "[REDACTED:email]"}
    assert safe["[REDACTED:email]"] == [None, True, 12, 1.5, ("safe",)]
    assert isinstance(references[0], RagChunk)
    assert references[0].text == "[REDACTED:email]"
    assert info == {
        "redacted": True,
        "redacted_count": 3,
        "redaction_counts": {"email": 3},
    }
    assert document.text == "admin@example.com"
    assert model.text == "admin@example.com"
    assert "extra@example.com" in structured


def test_redacted_model_extra_keys_are_sanitized_in_dictionary() -> None:
    """Sanitize model extra keys along with field values."""
    model = _Document(text="safe", **{"admin@example.com": "value"})
    _, safe, _, info = redact_tool_output(
        "safe", {"model": model}, None, MCP_RULES.compile(), 4000
    )
    assert safe["model"] == {"text": "safe", "[REDACTED:email]": "value"}
    assert info["redacted_count"] == 1


@pytest.mark.parametrize("document", [_ComputedDocument(), _SerializedDocument()])
def test_model_serialization_cannot_restore_redacted_metadata(
    document: BaseModel,
) -> None:
    """Keep computed fields and custom serializers out of the sanitized response."""
    original = document.model_dump()
    _, safe, _, info = redact_tool_output(
        "safe", {"document": document}, None, MCP_RULES.compile(), 4000
    )
    response = LLMResponse(
        conversation_id="review",
        response="safe",
        referenced_documents=[],
        truncated=False,
        input_tokens=0,
        output_tokens=0,
        available_quotas={},
        tool_calls=[],
        tool_results=[{"content": "safe", "structured_content": safe, **info}],
    )
    assert "admin@example.com" not in response.model_dump_json()
    assert safe == {"document": {"address": "[REDACTED:email]"}}
    assert info == {
        "redacted": True,
        "redacted_count": 1,
        "redaction_counts": {"email": 1},
    }
    assert document.model_dump() == original


@pytest.mark.parametrize(
    "content, expected_counts",
    [
        ("1415-555-01239", {"phone": 1}),
        ("1415-555-01239 admin@example.com", {"phone": 1, "email": 1}),
        ("1415-555-01239 212-555-0199", {"phone": 2}),
        ("415-555-0123", {"phone": 2}),
        ("1415-555-01239 415-555-0123", {"phone": 2}),
    ],
)
def test_metadata_counts_sum_actual_replacements(content, expected_counts) -> None:
    """Count each replacement, regardless of whether values also appear in text."""
    _, safe, _, info = redact_tool_output(
        content, {"phone": "415-555-0123"}, None, MCP_RULES.compile(), 4000
    )
    assert safe == {"phone": "[REDACTED:phone]"}
    assert info == {
        "redacted": True,
        "redacted_count": sum(expected_counts.values()),
        "redaction_counts": expected_counts,
    }


def test_metadata_counts_sum_all_rule_matches() -> None:
    """Add engine counts without subtracting unrelated text replacements."""
    _, _, _, info = redact_tool_output(
        "xpassword=alpha admin@example.com password=bravo",
        {"value": "password=alpha admin@example.com"},
        None,
        MCP_RULES.compile(),
        4000,
    )
    assert info["redaction_counts"] == {"email": 2, "credential": 2}
    assert info["redacted_count"] == 4


@pytest.mark.parametrize(
    "metadata,budget",
    [
        ({"payload": "x" * 1000}, 100),
        ({"payload": [None] * 1000}, 100),
        ({"unsupported": object()}, 100),
        ({"a@example.com": 1, "b@example.com": 2}, 4000),
    ],
)
def test_metadata_cannot_bypass_size_or_sanitization_guards(metadata, budget) -> None:
    """Reject oversized, unsupported, or ambiguously redacted metadata."""
    with pytest.raises(ValueError):
        redact_tool_output("safe", metadata, None, MCP_RULES.compile(), budget)
