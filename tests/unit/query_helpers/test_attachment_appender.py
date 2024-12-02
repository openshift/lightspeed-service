"""Unit tests for attachment appender."""

import pytest

from ols.app.models.models import Attachment
from ols.src.query_helpers.attachment_appender import (
    append_attachments_to_query,
    construct_intro_message,
    format_attachment,
    retrieve_kind_name_from_yaml,
)


@pytest.fixture
def test_yaml():
    """Proper YAML file for testing."""
    return """
kind: Pod
metadata:
     name: private-reg
"""


@pytest.fixture
def yaml_without_kind():
    """Proper YAML file without kind specified."""
    return """
metadata:
     name: private-reg
"""


@pytest.fixture
def yaml_without_name():
    """Proper YAML file without name specified."""
    return """
kind: Pod
metadata:
     foo: bar
"""


@pytest.fixture
def broken_yaml():
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


def test_format_attachment_in_yaml_format(test_yaml):
    """Test the function to format one attachment."""
    attachment = Attachment(
        attachment_type="log", content_type="application/yaml", content=test_yaml
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


def test_append_one_attachment_to_query(test_yaml):
    """Test how one attachment is appended to query."""
    attachments = [
        Attachment(
            attachment_type="config",
            content_type="application/yaml",
            content=test_yaml,
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


def test_append_two_attachments_to_query(test_yaml):
    """Test how two attachments are appended to query."""
    attachments = [
        Attachment(
            attachment_type="config",
            content_type="application/yaml",
            content=test_yaml,
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


def test_append_incomplete_yaml_query(yaml_without_kind, yaml_without_name):
    """Test how one attachment with incomplete YAML is appended to query."""
    attachments = [
        Attachment(
            attachment_type="config",
            content_type="application/yaml",
            content=yaml_without_kind,
        ),
    ]

    output = append_attachments_to_query("query", attachments)
    expected = """query

For reference, here is the full resource YAML:
```yaml

metadata:
     name: private-reg

```
"""
    assert output == expected

    attachments = [
        Attachment(
            attachment_type="config",
            content_type="application/yaml",
            content=yaml_without_name,
        ),
    ]

    output = append_attachments_to_query("query", attachments)
    expected = """query

For reference, here is the full resource YAML:
```yaml

kind: Pod
metadata:
     foo: bar

```
"""
    assert output == expected


def test_append_broken_yaml_query(broken_yaml):
    """Test how one attachment with broken YAML is appended to query."""
    attachments = [
        Attachment(
            attachment_type="config",
            content_type="application/yaml",
            content=broken_yaml,
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


def test_construct_intro_message_proper_yaml(test_yaml):
    """Check the function to construct intro message with proper YAML file at input."""
    message = construct_intro_message(test_yaml)
    expected = "For reference, here is the full resource YAML for Pod 'private-reg':"
    assert message == expected


def test_construct_intro_message_for_yaml_without_all_attributes(
    yaml_without_kind, yaml_without_name
):
    """Check the function to construct intro message with incomplete YAML file at input."""
    expected = "For reference, here is the full resource YAML:"

    message = construct_intro_message(yaml_without_kind)
    assert message == expected

    message = construct_intro_message(yaml_without_name)
    assert message == expected


def test_construct_intro_message_for_invalid_yaml(broken_yaml):
    """Check the function to construct intro message with broken YAML file at input."""
    expected = "For reference, here is the full resource YAML:"

    message = construct_intro_message(broken_yaml)
    assert message == expected


def test_retrieve_kind_name_from_yaml(test_yaml):
    """Check the function retrieve_kind_name_from_yaml for YAML with kind and name specified."""
    kind, name = retrieve_kind_name_from_yaml(test_yaml)
    assert kind == "Pod"
    assert name == "private-reg"


def test_retrieve_kind_name_from_yaml_without_kind(yaml_without_kind):
    """Check the function retrieve_kind_name_from_yaml for YAML file without kind specified."""
    kind, name = retrieve_kind_name_from_yaml(yaml_without_kind)
    assert kind is None
    assert name == "private-reg"


def test_retrieve_kind_name_from_yaml_without_name(yaml_without_name):
    """Check the function retrieve_kind_name_from_yaml for YAML file without name specified."""
    kind, name = retrieve_kind_name_from_yaml(yaml_without_name)
    assert kind == "Pod"
    assert name is None


def test_retrieve_kind_name_from__broken_yaml(broken_yaml):
    """Check the function retrieve_kind_name_from_yaml when broken YAML is used on input."""
    kind, name = retrieve_kind_name_from_yaml(broken_yaml)
    assert kind is None
    assert name is None
