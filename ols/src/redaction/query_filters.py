"""Adapt administrator-defined query filters to the shared redaction engine."""

from typing import cast

from ols.app.models.config import QueryFilter
from ols.src.redaction.engine import RedactionRule, RuleSet
from ols.src.redaction.engine import Redactor as EngineRedactor


class QueryFiltersRedactor:
    """Redact queries and attachments using precompiled administrator-defined filters."""

    def __init__(self, filters: list[QueryFilter] | None) -> None:
        """Compile configured filters in order without adding tool-output defaults."""
        self.rules = RuleSet(
            name="query-filters",
            rules=tuple(
                RedactionRule(
                    name=f"{index}:{query_filter.name}",
                    pattern=str(query_filter.pattern),
                    replacement=cast("str", query_filter.replace_with),
                )
                for index, query_filter in enumerate(filters or [])
            ),
            sequential=True,
        ).compile()

    def redact(self, conversation_id: str, text_input: str) -> str:
        """Redact text with the compiled query policy without logging payloads."""
        return EngineRedactor.redact(text_input, self.rules).text
