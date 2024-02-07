"""Unit tests for the configuration models."""

import io
import logging
import re
import traceback
from typing import TypeVar

import pytest
from pydantic import ValidationError

from ols.utils import config
from ols.utils.query_filter import RegexFilter

E = TypeVar("E", bound=Exception)


def check_expected_exception(
    yaml_stream: str, expected_exception_type: type[E], expected_error_msg: str
) -> None:
    """Check that an expected exception is raised."""
    with pytest.raises(expected_exception_type, match=expected_error_msg):
        config.load_config_from_stream(io.StringIO(yaml_stream))


def test_missing_config_file():
    """Check how missing configuration file is handled."""
    with pytest.raises(Exception, match=" Not a directory"):
        # /dev/null is special file so it can't be directory
        # at the same moment
        config.init_config("/dev/null/non-existent")


def test_invalid_config():
    """Check that invalid configuration is handled gracefully."""
    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    type: bam
    url: 'http://url1'
    models:
      - name: m1
        url: 'https://murl1'
      - name: m2
        url: 'https://murl2'
  - name: p2
    type: bam
    url: 'https://url2'
    models:
      - name: m1
        url: 'http://murl1'
      - name: m2
        url: 'http://murl2'
""",
        ValidationError,
        "ols_config\n  Field required",
    )

    check_expected_exception(
        """
---
llm_providers:
  bam:
    url: "https://bam-api.res.ibm.com"
    credentials_path: bam_api_key.txt
    models:
      - name: ibm/granite-13b-chat-v2
  openai:
    url: "htt1ps://api.openai.com/v1"
    credentials_path: openai_api_key.txt
    models:
      - name: gpt-4-1106-preview
      - name: gpt-3.5-turbo
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
  logging_config:
    app_log_level: info
    lib_log_level: warning
  default_provider: bam
  default_model: ibm/granite-13b-chat-v2
""",
        ValidationError,
        "URL scheme should be 'http' or 'https' ",
    )

    check_expected_exception(
        """
---
llm_providers:
  bam:
    url: "https://bam-api.res.ibm.com"
    credentials_path: bam_api_key.txt
    models:
      - name: 111
  openai:
    url: "https://api.openai.com/v1"
    credentials_path: openai_api_key.txt
    models:
      - name: gpt-4-1106-preview
      - name: gpt-3.5-turbo
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
  logging_config:
    app_log_level: info
    lib_log_level: warning
  default_provider: bam
  default_model: ibm/granite-13b-chat-v2
""",
        ValidationError,
        "name\n  Input should be a valid string",
    )

    for role in ["default"]:
        check_expected_exception(
            f"""
---
llm_providers:
  p1:
    type: bam
    url: 'http://url1'
    credentials_path: tests/config/secret.txt
    models:
      - name: m1
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
  logging_config:
    app_log_level: info
    lib_log_level: warning
  {role}_provider: bam
  {role}_model: m1
    """,
            ValidationError,
            f"{role}_provider 'bam' is not one of 'llm_providers'",
        )

        check_expected_exception(
            f"""
---
llm_providers:
  p1:
    type: bam
    url: 'http://url1'
    credentials_path: tests/config/secret.txt
    models:
      - name: m1
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
  logging_config:
    app_log_level: info
    lib_log_level: warning
  {role}_provider: p1
  {role}_model: no_such_model
    """,
            ValidationError,
            f"{role}_model 'no_such_model' is not in the models list for provider 'p1'",
        )

        check_expected_exception(
            f"""
---
llm_providers:
  p1:
    url: 'http://url1'
    credentials_path: openai_api_key.txt
    models:
      - name: m1
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
  logging_config:
    app_log_level: info
    lib_log_level: warning
  {role}_provider: p1
    """,
            ValidationError,
            "Both 'default_provider' and 'default_model' must be provided together or not at all",
        )

        check_expected_exception(
            f"""
---
llm_providers:
  p1:
    url: 'http://url1'
    credentials_path: openai_api_key.txt
    models:
      - name: m1
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
  logging_config:
    app_log_level: info
    lib_log_level: warning
  {role}_model: m1
    """,
            ValidationError,
            "Both 'default_provider' and 'default_model' must be provided together or not at all",
        )

    check_expected_exception(
        """
---
llm_providers:
  p1:
    url: 'http://url1'
    credentials_path: openai_api_key.txt
    models:
      - name: m1
ols_config:
  conversation_cache:
    foo: bar
  logging_config:
    app_log_level: info
    lib_log_level: warning
""",
        ValidationError,
        "conversation_cache.type\n  Field required",
    )

    check_expected_exception(
        """
---
llm_providers:
  p1:
    url: 'http://url1'
    credentials_path: openai_api_key.txt
    models:
      - name: m1
ols_config:
  conversation_cache:
    type: foobar
    redis:
      host: 127.0.0.1
      port: 1234
  logging_config:
    app_log_level: info
    lib_log_level: warning
""",
        ValidationError,
        "conversation_cache.type\n  Input should be 'redis' or 'memory'",
    )

    check_expected_exception(
        """
---
llm_providers:
  p1:
    url: 'http://url1'
    credentials_path: openai_api_key.txt
    models:
      - name: m1
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: foo
  logging_config:
""",
        ValidationError,
        "max_entries\n  Input should be a valid integer, unable to parse string as an integer",
    )

    check_expected_exception(
        """
---
llm_providers:
  p1:
    url: 'http://url1'
    credentials_path: openai_api_key.txt
    models:
      - name: m1
ols_config:
  conversation_cache:
    type: redis
    redis:
      max_memory_policy: foobar
  logging_config:
""",
        ValidationError,
        "Invalid Redis max_memory_policy: foobar, valid policies are ",
    )

    for port in ("-123", "1231231", "88888"):
        check_expected_exception(
            f"""
---
llm_providers:
  p1:
    url: 'http://url1'
    credentials_path: openai_api_key.txt
    models:
      - name: m1
ols_config:
  conversation_cache:
    type: redis
    redis:
      port: "{port}"
""",
            ValidationError,
            "Port number must be in 1-65535",
        )

    check_expected_exception(
        """
---
llm_providers:
  p1:
    url: 'http://url1'
    credentials_path: no_such_file_provider
    models:
      - name: m1
""",
        ValidationError,
        "Path does not point to a file",
    )

    check_expected_exception(
        """
---
llm_providers:
  p1:
    url: 'http://url1'
    credentials_path: openai_api_key.txt
    models:
      - name: m1
ols_config:
  reference_content:
    product_docs_index_id: "product"
    embeddings_model_path: "product"
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
dev_config:
  enable_dev_ui: true
  disable_auth: false

""",
        ValidationError,
        "product_docs_index_path\n  Field required",
    )


def test_valid_config_stream():
    """Check if a valid configuration stream is handled correctly."""
    try:
        config.load_config_from_stream(
            io.StringIO(
                """
---
llm_providers:
  p1:
    type: bam
    url: 'http://url1'
    credentials_path: tests/config/secret.txt
    models:
      - name: m1
      - name: m2
  p2:
    type: watsonx
    project_id: whatever
    url: 'https://url2'
    credentials_path: tests/config/secret.txt
    models:
      - name: m1
      - name: m2
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
  logging_config:
    level: info
  default_provider: p1
  default_model: m2
dev_config:
  enable_dev_ui: true
  disable_auth: false
  disable_tls: true
  llm_params:
    something: 5
"""
            )
        )
    except Exception as e:
        print(traceback.format_exc())
        pytest.fail(f"loading valid configuration failed: {e}")


def test_config_file_without_logging_config():
    """Check how a configuration file without logging config is correctly initialized."""
    # when logging configuration is not provided, default values will be used
    # it means the following call should not fail
    config.init_config("tests/config/config_without_logging.yaml")

    # test if default values have been set
    logging_config = config.ols_config.logging_config
    assert logging_config.app_log_level == logging.INFO
    assert logging_config.lib_log_level == logging.WARNING


# def test_valid_config_without_query_filter():
#     """Check if a valid configuration file without query filter creates empty regex filters."""
#     config.init_empty_config()
#     config.init_config("tests/config/valid_config_without_query_filter.yaml")
#     assert config.query_redactor is None
#     print(config.query_redactor)
#     config.init_query_filter()
#     assert config.query_redactor.regex_filters == []


def test_valid_config_with_query_filter():
    """Check if a valid configuration file with query filter is handled correctly."""
    config.query_redactor = None
    config.init_config("tests/config/valid_config_with_query_filter.yaml")
    config.init_query_filter()
    assert config.query_redactor.regex_filters == [
        RegexFilter(
            pattern=re.compile(r"\b(?:foo)\b"),
            name="foo_filter",
            replace_with="openshift",
        ),
        RegexFilter(
            pattern=re.compile(r"\b(?:bar)\b"),
            name="bar_filter",
            replace_with="kubernetes",
        ),
    ]
