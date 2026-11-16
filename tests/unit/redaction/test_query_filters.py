"""Unit tests for administrator-configured query redaction."""

import logging
import re
from unittest.mock import patch

import pytest

from ols.app.models.config import QueryFilter
from ols.src.redaction.engine import Redactor as EngineRedactor
from ols.src.redaction.query_filters import QueryFiltersRedactor


@pytest.mark.parametrize(
    "filters, query, expected",
    [
        (
            [
                {
                    "name": "word",
                    "pattern": r"\bimage\b",
                    "replace_with": "REDACTED_image",
                },
                {
                    "name": "ip",
                    "pattern": r"\d{1,3}(?:\.\d{1,3}){3}",
                    "replace_with": "REDACTED_IP",
                },
            ],
            "mongodb image with nodeip as 1.123.0.99",
            "mongodb REDACTED_image with nodeip as REDACTED_IP",
        ),
        (
            [
                {
                    "name": "partial",
                    "pattern": "mongo",
                    "replace_with": "REDACTED_MONGO",
                },
                {
                    "name": "url",
                    "pattern": r"(?:https?://)?(?:www\.)?[\w\.-]+\.\w+",
                    "replace_with": "",
                },
                {
                    "name": "phone",
                    "pattern": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                    "replace_with": "REDACTED_PHONE_NUMBER",
                },
            ],
            "mongodb from www.mongodb.com call 123-456-7890",
            "REDACTED_MONGOdb from  call REDACTED_PHONE_NUMBER",
        ),
        (
            [
                {"name": "same", "pattern": "Kubernetes", "replace_with": "FooBar"},
                {"name": "same", "pattern": "FooBar", "replace_with": "Baz"},
            ],
            "Tell me about Kubernetes",
            "Tell me about Baz",
        ),
        (
            [
                {
                    "name": "groups",
                    "pattern": r"(?i)(secret)=(\w+)",
                    "replace_with": r"\1=[HIDDEN]",
                }
            ],
            "SECRET=value",
            "SECRET=[HIDDEN]",
        ),
        ([], "admin@example.com 198.51.100.1", "admin@example.com 198.51.100.1"),
        (None, "admin@example.com 198.51.100.1", "admin@example.com 198.51.100.1"),
    ],
)
def test_query_filters_preserve_configured_behavior(filters, query, expected):
    """Preserve ordering, duplicate names, regex semantics and absent-filter behavior."""
    redactor = QueryFiltersRedactor(
        None if filters is None else [QueryFilter(item) for item in filters]
    )

    assert redactor.redact("conversation", query) == expected


def test_query_filters_use_engine_without_recompiling():
    """Compile at construction and reuse the engine for subsequent queries."""
    filters = [
        QueryFilter({"name": "word", "pattern": "secret", "replace_with": "hidden"})
    ]
    with patch(
        "ols.src.redaction.engine.re.compile", wraps=re.compile
    ) as compile_regex:
        redactor = QueryFiltersRedactor(filters)
        assert compile_regex.call_count == 1
        with patch.object(
            EngineRedactor, "redact", wraps=EngineRedactor.redact
        ) as redact:
            assert redactor.redact("first", "secret") == "hidden"
            assert redactor.redact("second", "secret secret") == "hidden hidden"
            assert redact.call_count == 2
        assert compile_regex.call_count == 1


def test_query_redaction_logs_do_not_disclose_input_or_filters(caplog):
    """Keep matched and unmatched query values and configuration out of logs."""
    with caplog.at_level(logging.DEBUG, logger="ols.src.redaction.query_filters"):
        redactor = QueryFiltersRedactor(
            [
                QueryFilter(
                    {
                        "name": "sensitive-filter-name",
                        "pattern": "private-token",
                        "replace_with": "hidden",
                    }
                )
            ]
        )
        assert (
            redactor.redact("conversation", "private-token unmatched-secret")
            == "hidden unmatched-secret"
        )

    assert "private-token" not in caplog.text
    assert "unmatched-secret" not in caplog.text
    assert "sensitive-filter-name" not in caplog.text
