"""Unit tests for attachment appender."""

from ols.app.models.models import Attachment
from ols.src.query_helpers.attachment_appender import (
    append_attachments_to_query,
    format_attachment,
)


def test_format_attachment_in_plain_text_format():
    """Test the function to format one attachment."""
    attachment = Attachment(
        attachment_type="log", content_type="text/plain", content="foo\nbar\nbaz"
    )
    formatted = format_attachment(attachment)

    expected = """
```
foo
bar
baz
```
"""
    assert formatted == expected


def test_format_attachment_in_yaml_format():
    """Test the function to format one attachment."""
    attachment = Attachment(
        attachment_type="log", content_type="application/yaml", content="foo\nbar\nbaz"
    )
    formatted = format_attachment(attachment)

    expected = """
```yaml
foo
bar
baz
```
"""
    assert formatted == expected


def test_append_attachment_to_query_on_no_attachments():
    """Test how list of empty attachments is handled."""
    output = append_attachments_to_query("query", [])
    expected = "query"
    assert output == expected


def test_append_one_attachment_to_query():
    """Test how one attachment is appended to query."""
    attachments = [
        Attachment(
            attachment_type="config",
            content_type="application/yaml",
            content="foo\nbar\nbaz",
        ),
    ]

    output = append_attachments_to_query("query", attachments)
    expected = """query
```yaml
foo
bar
baz
```
"""
    assert output == expected


def test_append_two_attachments_to_query():
    """Test how two attachments are appended to query."""
    attachments = [
        Attachment(
            attachment_type="config",
            content_type="application/yaml",
            content="foo\nbar\nbaz",
        ),
        Attachment(
            attachment_type="log",
            content_type="text/plain",
            content="foo\nbar\nbaz",
        ),
    ]

    output = append_attachments_to_query("query", attachments)
    expected = """query
```yaml
foo
bar
baz
```

```
foo
bar
baz
```
"""
    assert output == expected
