"""Unit tests for attachment appender."""

import pytest

from ols.app.models.models import Attachment
from ols.src.query_helpers.attachment_appender import (
    append_attachments_to_query,
    format_attachment,
    retrieve_kind_name_from_yaml,
)


@pytest.fixture
def _test_yaml():
    """Proper YAML file for testing."""
    return """
kind: Pod
metadata:
     name: private-reg
"""


@pytest.fixture()
def _broken_yaml():
    """Broken YAML file for testing."""
    return """
kindx: Pod
* metadata:
     name: private-reg
"""


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


def test_format_attachment_in_yaml_format(_test_yaml):
    """Test the function to format one attachment."""
    attachment = Attachment(
        attachment_type="log", content_type="application/yaml", content=_test_yaml
    )
    formatted = format_attachment(attachment)

    expected = """

For reference, here is the full resource YAML for Pod 'private-reg':
```yaml

kind: Pod
metadata:
     name: private-reg

```
"""

    assert formatted == expected


def test_append_attachment_to_query_on_no_attachments():
    """Test how list of empty attachments is handled."""
    output = append_attachments_to_query("query", [])
    expected = "query"
    assert output == expected


def test_append_one_attachment_to_query(_test_yaml):
    """Test how one attachment is appended to query."""
    attachments = [
        Attachment(
            attachment_type="config",
            content_type="application/yaml",
            content=_test_yaml,
        ),
    ]

    output = append_attachments_to_query("query", attachments)
    expected = """query

For reference, here is the full resource YAML for Pod 'private-reg':
```yaml

kind: Pod
metadata:
     name: private-reg

```
"""
    assert output == expected


def test_append_two_attachments_to_query(_test_yaml):
    """Test how two attachments are appended to query."""
    attachments = [
        Attachment(
            attachment_type="config",
            content_type="application/yaml",
            content=_test_yaml,
        ),
        Attachment(
            attachment_type="log",
            content_type="text/plain",
            content="foo\nbar\nbaz",
        ),
    ]

    output = append_attachments_to_query("query", attachments)
    expected = """query

For reference, here is the full resource YAML for Pod 'private-reg':
```yaml

kind: Pod
metadata:
     name: private-reg

```



```
foo
bar
baz
```
"""
    assert output == expected


def test_append_broken_yaml_query(_broken_yaml):
    """Test how one attachment with broken YAML is appended to query."""
    attachments = [
        Attachment(
            attachment_type="config",
            content_type="application/yaml",
            content=_broken_yaml,
        ),
    ]

    output = append_attachments_to_query("query", attachments)
    expected = """query

For reference, here is the full resource YAML:
```yaml

kindx: Pod
* metadata:
     name: private-reg

```
"""
    assert output == expected


def test_retrieve_kind_name_from_yaml(_test_yaml):
    """Check the function retrieve_kind_name_from_yaml."""
    kind, name = retrieve_kind_name_from_yaml(_test_yaml)
    assert kind == "Pod"
    assert name == "private-reg"


def test_retrieve_kind_name_from__broken_yaml(_broken_yaml):
    """Check the function retrieve_kind_name_from_yaml."""
    kind, name = retrieve_kind_name_from_yaml(_broken_yaml)
    assert kind is None
    assert name is None
