"""Sanitize bounded tool outputs and their structured projections."""

from collections import Counter
from dataclasses import fields, is_dataclass, replace
from typing import Any

from pydantic import BaseModel

from ols.src.redaction.engine import CompiledRuleSet, Redactor


class _MetadataRedactor:
    """Sanitize supported metadata values within a shared character/node budget."""

    def __init__(self, rules: CompiledRuleSet, budget: int) -> None:
        self.rules = rules
        self.remaining = budget
        self.counts: Counter[str] = Counter()

    def _charge(self, amount: int) -> None:
        self.remaining -= amount
        if self.remaining < 0:
            raise ValueError("tool metadata exceeds redaction budget")

    def _string(self, value: str) -> str:
        self._charge(len(value))
        result = Redactor.redact(value, self.rules)
        self.counts.update(result.counts)
        return result.text

    def _mapping(self, value: dict) -> dict:
        sanitized = {}
        for key, child in value.items():
            safe_key = self.visit(key)
            if safe_key in sanitized:
                raise ValueError("redaction produced duplicate mapping keys")
            sanitized[safe_key] = self.visit(child)
        return sanitized

    def visit(self, value: Any) -> Any:
        """Return a sanitized copy, rejecting unsupported or oversized values."""
        self._charge(1)
        sanitized: Any
        if value is None or isinstance(value, (bool, int, float)):
            sanitized = value
        elif isinstance(value, str):
            sanitized = self._string(value)
        elif isinstance(value, dict):
            sanitized = self._mapping(value)
        elif isinstance(value, list):
            sanitized = [self.visit(child) for child in value]
        elif isinstance(value, tuple):
            sanitized = tuple(self.visit(child) for child in value)
        elif isinstance(value, BaseModel):
            sanitized = self._mapping(value.model_dump())
        elif is_dataclass(value) and not isinstance(value, type):
            sanitized = replace(
                value,
                **{
                    field.name: self.visit(getattr(value, field.name))
                    for field in fields(value)
                },
            )
        else:
            raise ValueError("unsupported tool metadata type")
        return sanitized


def redact_tool_output(
    content: str,
    structured_content: dict | None,
    referenced_documents: list | None,
    rules: CompiledRuleSet,
    max_metadata_chars: int,
) -> tuple[str, dict | None, list | None, dict[str, Any]]:
    """Redact bounded output and metadata, summing engine replacement counts."""
    result = Redactor.redact(content, rules)
    visitor = _MetadataRedactor(rules, max_metadata_chars)
    structured = (
        visitor.visit(structured_content) if structured_content is not None else None
    )
    references = (
        visitor.visit(referenced_documents)
        if referenced_documents is not None
        else None
    )
    counts = Counter(result.counts) + visitor.counts
    metadata: dict[str, Any] = {}
    if counts:
        metadata = {
            "redacted": True,
            "redacted_count": sum(counts.values()),
            "redaction_counts": dict(counts),
        }
    return result.text, structured, references, metadata
