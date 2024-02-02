"""Unit tests for the configuration models."""

import io
import logging
import traceback
from typing import TypeVar

import pytest
from yaml.parser import ParserError

from ols import constants
from ols.app.models.config import Config, InvalidConfigurationError
from ols.utils import config

E = TypeVar("E", bound=Exception)


def check_expected_exception(
    yaml_stream: str, expected_exception_type: type[E], expected_error_msg: str
) -> None:
    """Check that an expected exception is raised."""
    with pytest.raises(expected_exception_type, match=expected_error_msg):
        config.load_config_from_stream(io.StringIO(yaml_stream))


def test_malformed_yaml():
    """Check that malformed YAML is handled gracefully."""
    check_expected_exception(
        """[foo=123}""", ParserError, "while parsing a flow sequence"
    )
    check_expected_exception(
        """foobar""", AttributeError, "'str' object has no attribute 'get'"
    )
    check_expected_exception(
        """123""", AttributeError, "'int' object has no attribute 'get'"
    )
    check_expected_exception(
        """12.3""", AttributeError, "'float' object has no attribute 'get'"
    )
    check_expected_exception(
        """[1,2,3]""", AttributeError, "'list' object has no attribute 'get'"
    )


def test_missing_config_file():
    """Check how missing configuration file is handled."""
    with pytest.raises(Exception, match="Not a directory"):
        # /dev/null is special file so it can't be directory
        # at the same moment
        config.init_config("/dev/null/non-existent")


def test_invalid_config():
    """Check that invalid configuration is handled gracefully."""
    check_expected_exception(
        """""", InvalidConfigurationError, "no LLM providers config section found"
    )
    check_expected_exception(
        """{foo=123}""",
        InvalidConfigurationError,
        "no LLM providers config section found",
    )
    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    url: 'http://url1'
    models:
      - name: m1
        url: 'https://murl1'
      - name: m2
        url: 'https://murl2'
  - name: p2
    url: 'https://url2'
    models:
      - name: m1
        url: 'http://murl1'
      - name: m2
        url: 'http://murl2'
""",
        InvalidConfigurationError,
        "no OLS config section found",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    url: 'http://url1'
    models:
      - name: m1
        url: 'this-is-not-valid-url'
      - name: m2
        url: 'https://murl2'
  - name: p2
    url: 'https://url2'
    models:
      - name: m1
        url: 'http://murl1'
      - name: m2
        url: 'http://murl2'
""",
        InvalidConfigurationError,
        "model URL is invalid",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    url: 'not-valid-url'
    models:
      - name: m1
        url: 'https://murl1'
      - name: m2
        url: 'https://murl2'
  - name: p2
    url: 'https://url2'
    models:
      - name: m1
        url: 'http://murl1'
      - name: m2
        url: 'http://murl2'
""",
        InvalidConfigurationError,
        "provider URL is invalid",
    )
    check_expected_exception(
        """
---
llm_providers:
  - foo: p1
    url: 'http://url1'
    models:
      - name: m1
        url: 'http://murl1'
""",
        InvalidConfigurationError,
        "provider name is missing",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    url: foobar
    models:
      - name: m1
        url: 'http://murl1'
""",
        InvalidConfigurationError,
        "provider URL is invalid",
    )

    for role in constants.PROVIDER_MODEL_ROLES:
        check_expected_exception(
            f"""
---
llm_providers:
  - name: p1
    url: 'http://url1'
    models:
      - name: m1
        url: 'http://murl1'
ols_config:
  {role}_provider: no_such_provider
  {role}_model: m1
    """,
            InvalidConfigurationError,
            f"{role}_provider specifies an unknown provider no_such_provider",
        )

        check_expected_exception(
            f"""
---
llm_providers:
  - name: p1
    url: 'http://url1'
    models:
      - name: m1
        url: 'http://murl1'
ols_config:
  {role}_provider: p1
  {role}_model: no_such_model
    """,
            InvalidConfigurationError,
            f"{role}_model specifies an unknown model no_such_model",
        )

        check_expected_exception(
            f"""
---
llm_providers:
  - name: p1
    url: 'http://url1'
    models:
      - name: m1
        url: 'http://murl1'
ols_config:
  {role}_provider: p1
    """,
            InvalidConfigurationError,
            f"{role}_provider is specified, but {role}_model is missing",
        )

        check_expected_exception(
            f"""
---
llm_providers:
  - name: p1
    url: 'http://url1'
    models:
      - name: m1
        url: 'http://murl1'
ols_config:
  {role}_model: m1
    """,
            InvalidConfigurationError,
            f"{role}_model is specified, but {role}_provider is missing",
        )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    url: 'http://url1'
    models:
      - name: m1
        url: 'https://murl1'
ols_config:
  conversation_cache:
    type: memory
    redis:
      url: 127.0.0.1:1234
      credentials:
          user: root
          password: pwd123
""",
        InvalidConfigurationError,
        "memory configuration is missing",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    url: 'http://url1'
    models:
      - name: m1
        url: 'https://murl1'
ols_config:
  conversation_cache:
    type: redis
    memory:
      max_entries: 1000
""",
        InvalidConfigurationError,
        "redis configuration is missing",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    url: 'http://url1'
    credentials_path: no_such_file_provider
""",
        FileNotFoundError,
        "No such file or directory: 'no_such_file_provider'",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    url: 'http://url1'
    credentials_path: tests/config/secret.txt
    models:
      - name: m1
        url: 'https://murl1'
        credentials_path: no_such_file_model
""",
        FileNotFoundError,
        "No such file or directory: 'no_such_file_model'",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    credentials_path: tests/config/secret.txt
    models:
      - name: m1
        credentials_path: tests/config/secret.txt
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
dev_config:
  llm_temperature_override: NaN
""",
        InvalidConfigurationError,
        "llm_temperature_override must be a float",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    credentials_path: tests/config/secret.txt
    models:
      - name: m1
        credentials_path: tests/config/secret.txt
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
dev_config:
  llm_temperature_override: -1
""",
        InvalidConfigurationError,
        "llm_temperature_override must be between 0 and 1",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    credentials_path: tests/config/secret.txt
    models:
      - name: m1
        credentials_path: tests/config/secret.txt
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
dev_config:
  llm_temperature_override: 1.1
  enable_dev_ui: true
  disable_question_validation: false
  disable_auth: false

""",
        InvalidConfigurationError,
        "llm_temperature_override must be between 0 and 1",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    credentials_path: tests/config/secret.txt
    models:
      - name: m1
        credentials_path: tests/config/secret.txt
ols_config:
  reference_content:
    product_docs_index_path: "./invalid_dir"
    product_docs_index_id: product
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
dev_config:
  llm_temperature_override: 0.1
  enable_dev_ui: true
  disable_question_validation: false
  disable_auth: false

""",
        InvalidConfigurationError,
        "Reference content path './invalid_dir' does not exist",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    credentials_path: tests/config/secret.txt
    models:
      - name: m1
        credentials_path: tests/config/secret.txt
ols_config:
  reference_content:
    product_docs_index_path: "/tmp"
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
dev_config:
  llm_temperature_override: 0.1
  enable_dev_ui: true
  disable_question_validation: false
  disable_auth: false

""",
        InvalidConfigurationError,
        "product_docs_index_path is specified but product_docs_index_id is missing",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    credentials_path: tests/config/secret.txt
    models:
      - name: m1
        credentials_path: tests/config/secret.txt
ols_config:
  reference_content:
    product_docs_index_id: "product"
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
dev_config:
  llm_temperature_override: 0.1
  enable_dev_ui: true
  disable_question_validation: false
  disable_auth: false

""",
        InvalidConfigurationError,
        "product_docs_index_id is specified but product_docs_index_path is missing",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    credentials_path: tests/config/secret.txt
    models:
      - name: m1
        credentials_path: tests/config/secret.txt
ols_config:
  reference_content:
    product_docs_index_path: "tests/config/secret.txt"
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
dev_config:
  llm_temperature_override: 0.1
  enable_dev_ui: true
  disable_question_validation: false
  disable_auth: false

""",
        InvalidConfigurationError,
        "Reference content path 'tests/config/secret.txt' is not a directory",
    )


def test_valid_config_stream():
    """Check if a valid configuration stream is handled correctly."""
    try:
        config.load_config_from_stream(
            io.StringIO(
                """
---
llm_providers:
  - name: p1
    url: 'http://url1'
    credentials_path: tests/config/secret.txt
    models:
      - name: m1
        url: 'http://murl1'
        credentials_path: tests/config/secret.txt
      - name: m2
        url: 'https://murl2'
  - name: p2
    url: 'https://url2'
    models:
      - name: m1
        url: 'https://murl1'
      - name: m2
        url: 'https://murl2'
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
  logging_config:
    level: info
  default_provider: p1
  default_model: m1
  classifier_provider: p2
  classifier_model: m1
dev_config:
  llm_temperature_override: 0
  enable_dev_ui: true
  disable_question_validation: false
  disable_auth: false
"""
            )
        )
    except Exception:
        print(traceback.format_exc())
        pytest.fail()


def test_valid_config_file():
    """Check if a valid configuration file is handled correctly."""
    try:
        config.init_config("tests/config/valid_config.yaml")

        expected_config = Config(
            {
                "llm_providers": [
                    {
                        "name": "p1",
                        "url": "https://url1",
                        "credentials_path": "tests/config/secret.txt",
                        "models": [
                            {
                                "name": "m1",
                                "url": "https://murl1",
                                "credentials_path": "tests/config/secret.txt",
                            },
                            {
                                "name": "m2",
                                "url": "https://murl2",
                            },
                        ],
                    },
                    {
                        "name": "p2",
                        "url": "https://url2",
                        "models": [
                            {
                                "name": "m1",
                                "url": "https://murl1",
                            },
                            {
                                "name": "m2",
                                "url": "https://murl2",
                            },
                        ],
                    },
                ],
                "ols_config": {
                    "reference_content": {
                        "product_docs_index_path": "tests/config",
                        "product_docs_index_id": "product",
                    },
                    "conversation_cache": {
                        "type": "memory",
                        "memory": {
                            "max_entries": 1000,
                        },
                    },
                    "logging_config": {
                        "logging_level": "INFO",
                    },
                    "default_provider": "p1",
                    "default_model": "m1",
                    "classifier_provider": "p1",
                    "classifier_model": "m2",
                    "summarizer_provider": "p1",
                    "summarizer_model": "m1",
                    "validator_provider": "p1",
                    "validator_model": "m1",
                },
            }
        )
        assert config.config == expected_config
    except Exception:
        print(traceback.format_exc())
        pytest.fail()


def test_valid_config_file_with_redis():
    """Check if a valid configuration file with Redis conversation cache is handled correctly."""
    try:
        config.init_config("tests/config/valid_config_redis.yaml")

        expected_config = Config(
            {
                "llm_providers": [
                    {
                        "name": "p1",
                        "url": "https://url1",
                        "credentials_path": "tests/config/secret.txt",
                        "models": [
                            {
                                "name": "m1",
                                "url": "https://murl1",
                            },
                        ],
                    },
                ],
                "ols_config": {
                    "reference_content": {
                        "product_docs_index_path": "tests/config",
                        "product_docs_index_id": "product",
                    },
                    "conversation_cache": {
                        "type": "redis",
                        "redis": {
                            "host": "foobar.com",
                            "port": "1234",
                            "max_memory": "100MB",
                            "max_memory_policy": "allkeys-lru",
                            "credentials": {
                                "user_path": "tests/config/redis_user.txt",
                                "password_path": "tests/config/redis_password.txt",
                            },
                        },
                    },
                    "logging_config": {
                        "logging_level": "INFO",
                    },
                    "default_provider": "p1",
                    "default_model": "m1",
                },
            }
        )
        assert config.config == expected_config
    except Exception:
        print(traceback.format_exc())
        pytest.fail()


def test_config_file_without_logging_config():
    """Check how a configuration file without logging config is correctly initialized."""
    # when logging configuration is not provided, default values will be used
    # it means the following call should not fail
    config.init_config("tests/config/config_without_logging.yaml")

    # test if default values have been set
    logging_config = config.ols_config.logging_config
    assert logging_config.app_log_level == logging.INFO
    assert logging_config.lib_log_level == logging.WARNING
