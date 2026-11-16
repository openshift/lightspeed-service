"""Benchmarks for precompiled administrator-defined query filters."""

import pytest

from ols.app.models.config import QueryFilter
from ols.src.redaction.query_filters import QueryFiltersRedactor

FILTERS = (
    ("any_string_match", r"(?:mongo)", "REDACTED_MONGO"),
    ("url", r"(?:https?://)?(?:www\.)?[\w\.-]+\.\w+", ""),
    ("phone_number", r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "REDACTED_PHONE_NUMBER"),
    ("perfect_word", r"\b(?:image)\b", "REDACTED_image"),
    ("ip_address", r"(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", "REDACTED_IP"),
)


@pytest.mark.parametrize(
    "filters, query",
    [
        ((), ""),
        ((), "write a deployment yaml for the mongodb image"),
        (
            FILTERS[3:],
            "write a deployment yaml for the mongodb image with nodeip as 1.123.0.99",
        ),
        (FILTERS[:3], "mongodb image from www.mongodb.com and call me at 123-456-7890"),
        (FILTERS, "mongodb image from www.mongodb.com and call me at 123-456-7890"),
        (
            FILTERS * 20,
            "mongodb image from www.mongodb.com and call me at 123-456-7890",
        ),
    ],
    ids=[
        "empty-query",
        "no-filters",
        "image-ip",
        "url-phone",
        "five-filters",
        "100-filters",
    ],
)
def test_query_filter_redaction(benchmark, filters, query):
    """Measure application of compiled query policies without compilation overhead."""
    redactor = QueryFiltersRedactor(
        [
            QueryFilter({"name": name, "pattern": pattern, "replace_with": replacement})
            for name, pattern, replacement in filters
        ]
    )
    benchmark(redactor.redact, "test_id", query)
