"""Bencharks for attachment appender."""

# pylint: disable=W0621

import pytest

from ols.app.models.models import Attachment
from ols.src.query_helpers.attachment_appender import (
    append_attachments_to_query,
    format_attachment,
)


@pytest.fixture
def short_query():
    """Short query for benchmarks."""
    return "What is Kubernetes?"


@pytest.fixture
def long_query():
    """Long query for benchmarks."""
    return "What is Kubernetes? " * 1000


@pytest.fixture
def yaml_content():
    """Proper YAML file for testing."""
    return """
kind: Pod
metadata:
     name: private-reg
"""


@pytest.fixture
def long_yaml_content():
    """Proper YAML file for testing."""
    numbers = ""
    for i in range(10000):
        numbers += f"- Number_{i}\n"

    return """
kind: Pod
metadata:
     name: private-reg
foods:
- Apple
- Orange
- Strawberry
- Mango
numbers:
""" + numbers


def test_format_attachment_empty_text_plain_format(benchmark):
    """Benchmark the function to format one attachment that is empty."""
    attachment = Attachment(
        attachment_type="log", content_type="text/plain", content=""
    )
    benchmark(format_attachment, attachment)


def test_format_attachment_empty_yaml_format(benchmark):
    """Benchmark the function to format one attachment that is empty."""
    attachment = Attachment(
        attachment_type="log", content_type="application/yaml", content=""
    )
    benchmark(format_attachment, attachment)


def test_format_attachment_plain_text_format(benchmark):
    """Benchmark the function to format one attachment in plain text format."""
    attachment = Attachment(
        attachment_type="log", content_type="text/plain", content="foo\nbar\nbaz"
    )
    benchmark(format_attachment, attachment)


def test_format_long_attachment_plain_text_format(benchmark):
    """Benchmark the function to format one attachment in plain text format."""
    content = "foo\nbar\nbaz" * 10000
    attachment = Attachment(
        attachment_type="log", content_type="text/plain", content=content
    )
    benchmark(format_attachment, attachment)


def test_format_attachment_yaml(yaml_content, benchmark):
    """Benchmark the function to format one attachment in YAML format."""
    attachment = Attachment(
        attachment_type="log", content_type="application/yaml", content=yaml_content
    )
    benchmark(format_attachment, attachment)


def test_format_long_attachment_yaml(long_yaml_content, benchmark):
    """Benchmark the function to format one attachment in YAML format."""
    attachment = Attachment(
        attachment_type="log",
        content_type="application/yaml",
        content=long_yaml_content,
    )
    benchmark(format_attachment, attachment)


def test_append_attachments_to_query_one_attachment(benchmark, short_query):
    """Benchmark the function to append attachments to query."""
    content = "foo\nbar\nbaz"
    attachment = Attachment(
        attachment_type="log", content_type="text/plain", content=content
    )
    benchmark(append_attachments_to_query, short_query, [attachment])


def test_append_attachments_to_query_3_attachments(
    benchmark, short_query, yaml_content
):
    """Benchmark the function to append attachments to query."""
    content = "foo\nbar\nbaz"
    attachment1 = Attachment(
        attachment_type="log", content_type="text/plain", content=content
    )
    attachment2 = Attachment(
        attachment_type="log", content_type="text/plain", content=content
    )
    attachment3 = Attachment(
        attachment_type="log",
        content_type="application/yaml",
        content=yaml_content,
    )
    benchmark(
        append_attachments_to_query,
        short_query,
        [attachment1, attachment2, attachment3],
    )


def test_append_attachments_to_query_100_attachments(benchmark, short_query):
    """Benchmark the function to append attachments to query."""
    content = "foo\nbar\nbaz"
    attachment1 = Attachment(
        attachment_type="log", content_type="text/plain", content=content
    )
    attachment2 = Attachment(
        attachment_type="log", content_type="text/plain", content=content
    )
    attachments = [attachment1, attachment2] * 50
    benchmark(append_attachments_to_query, short_query, attachments)


def test_append_attachments_to_query_long_query(benchmark, long_query):
    """Benchmark the function to append attachments to query."""
    content = "foo\nbar\nbaz"
    attachment = Attachment(
        attachment_type="log", content_type="text/plain", content=content
    )
    benchmark(append_attachments_to_query, long_query, [attachment])
