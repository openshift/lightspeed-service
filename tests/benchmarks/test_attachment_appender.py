"""Bencharks for attachment appender."""

import pytest

from ols.app.models.models import Attachment
from ols.src.query_helpers.attachment_appender import format_attachment


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

    return (
        """
kind: Pod
metadata:
     name: private-reg
foods:
- Apple
- Orange
- Strawberry
- Mango
numbers:
"""
        + numbers
    )


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
