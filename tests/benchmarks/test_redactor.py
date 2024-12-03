"""Benchmarks for the question filter."""

# pylint: disable=W0621

import re

import pytest

from ols.utils.config import AppConfig
from ols.utils.redactor import Redactor, RegexFilter


@pytest.fixture
def config():
    """Load configuration used for benchmarks."""
    config = AppConfig()
    config.reload_from_yaml_file("tests/config/valid_config_with_query_filter.yaml")
    return config


def test_redact_query_with_empty_query(benchmark, config):
    """Benchmark redact query with empty filters."""
    query_filter = Redactor(config.ols_config.query_filters)
    query = ""
    query_filter.regex_filters = []
    benchmark(query_filter.redact, "test_id", query)


def test_redact_query_with_empty_filters(benchmark, config):
    """Benchmark redact query with empty filters."""
    query_filter = Redactor(config.ols_config.query_filters)
    query = "write a deployment yaml for the mongodb image"
    query_filter.regex_filters = []
    benchmark(query_filter.redact, "test_id", query)


def test_redact_question_image_ip(benchmark, config):
    """Benchmark redact question with perfect word and ip."""
    query_filter = Redactor(config.ols_config.query_filters)
    query_filter.regex_filters = [
        RegexFilter(re.compile(r"\b(?:image)\b"), "perfect_word", "REDACTED_image"),
        RegexFilter(
            re.compile(r"(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"),
            "ip_address",
            "REDACTED_IP",
        ),
    ]
    query = "write a deployment yaml for the mongodb image with nodeip as 1.123.0.99"
    benchmark(query_filter.redact, "test_id", query)


def test_redact_question_mongopart_url_phone(benchmark, config):
    """Benchmark redact question with partial_word, url and phone number."""
    query_filter = Redactor(config.ols_config.query_filters)
    query_filter.regex_filters = [
        RegexFilter(re.compile(r"(?:mongo)"), "any_string_match", "REDACTED_MONGO"),
        RegexFilter(
            re.compile(r"(?:https?://)?(?:www\.)?[\w\.-]+\.\w+"),
            "url",
            "",
        ),
        RegexFilter(
            re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
            "phone_number",
            "REDACTED_PHONE_NUMBER",
        ),
    ]
    query = "write a deployment yaml for\
    the mongodb image from www.mongodb.com and call me at 123-456-7890"
    benchmark(query_filter.redact, "test_id", query)


def test_redact_question_multiple_filters(benchmark, config):
    """Benchmark question redaction by multiple filters."""
    query_filter = Redactor(config.ols_config.query_filters)
    query_filter.regex_filters = [
        RegexFilter(re.compile(r"(?:mongo)"), "any_string_match", "REDACTED_MONGO"),
        RegexFilter(
            re.compile(r"(?:https?://)?(?:www\.)?[\w\.-]+\.\w+"),
            "url",
            "",
        ),
        RegexFilter(
            re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
            "phone_number",
            "REDACTED_PHONE_NUMBER",
        ),
        RegexFilter(re.compile(r"\b(?:image)\b"), "perfect_word", "REDACTED_image"),
        RegexFilter(
            re.compile(r"(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"),
            "ip_address",
            "REDACTED_IP",
        ),
    ]
    query = "write a deployment yaml for\
    the mongodb image from www.mongodb.com and call me at 123-456-7890"
    benchmark(query_filter.redact, "test_id", query)


def test_redact_question_100_filters(benchmark, config):
    """Test question redaction by using 100 filters."""
    query_filter = Redactor(config.ols_config.query_filters)
    query_filter.regex_filters = [
        RegexFilter(re.compile(r"(?:mongo)"), "any_string_match", "REDACTED_MONGO"),
        RegexFilter(
            re.compile(r"(?:https?://)?(?:www\.)?[\w\.-]+\.\w+"),
            "url",
            "",
        ),
        RegexFilter(
            re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
            "phone_number",
            "REDACTED_PHONE_NUMBER",
        ),
        RegexFilter(re.compile(r"\b(?:image)\b"), "perfect_word", "REDACTED_image"),
        RegexFilter(
            re.compile(r"(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"),
            "ip_address",
            "REDACTED_IP",
        ),
    ] * 20
    query = "write a deployment yaml for\
    the mongodb image from www.mongodb.com and call me at 123-456-7890"
    benchmark(query_filter.redact, "test_id", query)
