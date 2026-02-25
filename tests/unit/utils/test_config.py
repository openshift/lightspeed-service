"""Unit tests for the configuration models."""

import io
import logging
import re
import traceback
from typing import TypeVar
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from yaml.parser import ParserError

from ols import config, constants
from ols.app.models.config import Config
from ols.utils.checks import InvalidConfigurationError
from ols.utils.redactor import RegexFilter

E = TypeVar("E", bound=Exception)


def check_expected_exception(
    yaml_stream: str, expected_exception_type: type[E], expected_error_msg: str
) -> None:
    """Check that an expected exception is raised."""
    with pytest.raises(expected_exception_type, match=expected_error_msg):
        config._load_config_from_yaml_stream(io.StringIO(yaml_stream))


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
        config.reload_from_yaml_file("/dev/null/non-existent")


def test_invalid_dev_config():
    """Check that invalid configuration is handled gracefully."""
    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    type: bam
    credentials_path: tests/config/secret/apitoken
    models:
      - name: m1
        credentials_path: tests/config/secret/apitoken
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
dev_config:
  llm_params:
     - something: 0
""",
        ValidationError,
        "Input should be a valid dictionary",
    )


def test_invalid_config_missing_llm_providers_section():
    """Check handling invalid configuration without LLM providers section."""
    check_expected_exception(
        """
---
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
""",
        InvalidConfigurationError,
        "no LLM providers config section found",
    )


def test_invalid_config_missing_ols_config_section():
    """Check handling invalid configuration without OLS section."""
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
        InvalidConfigurationError,
        "no OLS config section found",
    )


def test_invalid_config_invalid_model_url():
    """Check handling invalid configuration containing invalid model URL."""
    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    type: bam
    url: 'http://url1'
    models:
      - name: m1
        url: 'this-is-not-valid-url'
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
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
""",
        ValidationError,
        "Input should be a valid URL, relative URL without a base",
    )


def test_invalid_config_invalid_provider_url():
    """Check handling invalid configuration containing invalid provider URL."""
    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    type: bam
    url: 'not-valid-url'
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
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
""",
        InvalidConfigurationError,
        "provider URL is invalid",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    type: bam
    url: foobar
    models:
      - name: m1
        url: 'http://murl1'
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
""",
        InvalidConfigurationError,
        "provider URL is invalid",
    )


def test_invalid_config_missing_provider_name():
    """Check handling invalid configuration without provider name."""
    check_expected_exception(
        """
---
llm_providers:
  - foo: p1
    type: bam
    url: 'http://url1'
    models:
      - name: m1
        url: 'http://murl1'
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
""",
        InvalidConfigurationError,
        "provider name is missing",
    )


def test_invalid_config_unknown_provider_name():
    """Check handling invalid configuration having unknown provider name."""
    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    type: bam
    url: 'http://url1'
    models:
      - name: m1
        url: 'http://murl1'
dev_config:
  disable_tls: true
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
  default_provider: no_such_provider
  default_model: m1
    """,
        InvalidConfigurationError,
        "default_provider specifies an unknown provider no_such_provider",
    )


def test_invalid_config_unknown_model_name():
    """Check handling invalid configuration having unknown model name."""
    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    type: bam
    url: 'http://url1'
    models:
      - name: m1
        url: 'http://murl1'
dev_config:
  disable_tls: true
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
  default_provider: p1
  default_model: no_such_model
    """,
        InvalidConfigurationError,
        "default_model specifies an unknown model no_such_model",
    )


def test_invalid_config_missing_default_model():
    """Check handling invalid configuration without default model."""
    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    type: bam
    url: 'http://url1'
    models:
      - name: m1
        url: 'http://murl1'
dev_config:
  disable_tls: true
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
  default_provider: p1
    """,
        InvalidConfigurationError,
        "default_model is missing",
    )


def test_invalid_config_missing_default_provider():
    """Check handling invalid configuration without default provider."""
    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    type: bam
    url: 'http://url1'
    models:
      - name: m1
        url: 'http://murl1'
dev_config:
  disable_tls: true
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
  default_model: m1
    """,
        InvalidConfigurationError,
        "default_provider is missing",
    )


def test_invalid_config_for_memory_cache():
    """Check handling invalid memory cache configuration."""
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
dev_config:
  disable_tls: true
ols_config:
  conversation_cache:
    type: memory
    redis:
      host: 127.0.0.1
      port: 1234
      password_path: pwd123
""",
        InvalidConfigurationError,
        "memory conversation cache type is specified,"
        " but memory configuration is missing",
    )

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
dev_config:
  disable_tls: true
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: foo
""",
        InvalidConfigurationError,
        "invalid max_entries for memory conversation cache,"
        " max_entries needs to be a non-negative integer",
    )


def test_invalid_config_improper_credentials():
    """Test invalid config with improper credentials."""
    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    type: bam
    url: 'http://url1'
    credentials_path: no_such_file
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
""",
        FileNotFoundError,
        "No such file or directory: 'no_such_file'",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    type: bam
    url: 'http://url1'
    credentials_path: tests/config/secret/apitoken
    models:
      - name: m1
        url: 'https://murl1'
        credentials_path: no_such_file
ols_config:
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
""",
        FileNotFoundError,
        "No such file or directory: 'no_such_file'",
    )


def test_invalid_config_improper_reference_content():
    """Test invalid config with improper reference content."""
    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    type: bam
    credentials_path: tests/config/secret/apitoken
    models:
      - name: m1
        credentials_path: tests/config/secret/apitoken
ols_config:
  reference_content:
    indexes:
    - product_docs_index_path: "./invalid_dir"
      product_docs_index_id: product
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
dev_config:
  enable_dev_ui: true
  disable_auth: false
  pyroscope_url: https://pyroscope.pyroscope.svc.cluster.local:4040

""",
        InvalidConfigurationError,
        "Reference content index path './invalid_dir' does not exist",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    type: bam
    credentials_path: tests/config/secret/apitoken
    models:
      - name: m1
        credentials_path: tests/config/secret/apitoken
ols_config:
  reference_content:
    embeddings_model_path: ./invalid_dir
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
dev_config:
  llm_temperature_override: 0.1
  enable_dev_ui: true
  disable_auth: false
  pyroscope_url: https://pyroscope.pyroscope.svc.cluster.local:4040

""",
        InvalidConfigurationError,
        "Embeddings model path './invalid_dir' does not exist",
    )

    check_expected_exception(
        """
---
llm_providers:
  - name: p1
    type: bam
    credentials_path: tests/config/secret/apitoken
    models:
      - name: m1
        credentials_path: tests/config/secret/apitoken
ols_config:
  reference_content:
    indexes:
    - product_docs_index_id: "product"
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
dev_config:
  enable_dev_ui: true
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
    type: bam
    credentials_path: tests/config/secret/apitoken
    models:
      - name: m1
        credentials_path: tests/config/secret/apitoken
ols_config:
  reference_content:
    indexes:
    - product_docs_index_path: "tests/config/secret/apitoken"
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
dev_config:
  enable_dev_ui: true
  disable_auth: false

""",
        InvalidConfigurationError,
        "Reference content index path 'tests/config/secret/apitoken' is not a directory",
    )


def test_unreadable_directory():
    """Check if an unredable directory is reported correctly."""
    with patch("os.access", return_value=False):
        check_expected_exception(
            """
---
llm_providers:
  - name: p1
    type: bam
    credentials_path: tests/config/secret/apitoken
    models:
      - name: m1
        credentials_path: tests/config/secret/apitoken
ols_config:
  reference_content:
    embeddings_model_path: tests/config
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
dev_config:
  llm_temperature_override: 0.1
  enable_dev_ui: true
  disable_auth: false
  disable_tls: true

""",
            InvalidConfigurationError,
            "Embeddings model path 'tests/config' is not readable",
        )


def test_valid_config_stream():
    """Check if a valid configuration stream is handled correctly."""
    try:
        config._load_config_from_yaml_stream(io.StringIO("""
---
llm_providers:
  - name: p1
    type: bam
    url: 'http://url1'
    credentials_path: tests/config/secret/apitoken
    models:
      - name: m1
        url: 'http://murl1'
        credentials_path: tests/config/secret/apitoken
      - name: m2
        url: 'https://murl2'
    tlsSecurityProfile:
      type: Custom
      ciphers:
          - TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
          - TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
      minTLSVersion: VersionTLS13
  - name: p2
    type: bam
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
  quota_handlers:
    storage:
      host: ""
      port: "5432"
      dbname: "test"
      user: "tester"
      password_path: tests/config/postgres_password.txt
      ssl_mode: "disable"
    limiters:
      - name: user_monthly_limits
        type: user_limiter
        initial_quota: 1000
        quota_increase: 10
        period: "5 minutes"
      - name: cluster_monthly_limits
        type: cluster_limiter
        initial_quota: 2000
        quota_increase: 100
        period: "5 minutes"
    scheduler:
      period: 100
  certificate_directory: '/foo/bar/baz'
  system_prompt_path: 'tests/config/system_prompt.txt'
  tlsSecurityProfile:
    type: Custom
    ciphers:
        - TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
        - TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
    minTLSVersion: VersionTLS13
mcp_servers:
  - name: foo
    url: http://foo-server:8080/mcp
  - name: bar
    url: http://bar-server:8080/mcp
dev_config:
  enable_dev_ui: true
  disable_auth: false
  disable_tls: true
  llm_params:
    something: 5
"""))
    except Exception as e:
        print(traceback.format_exc())
        pytest.fail(f"loading valid configuration failed: {e}")


def test_valid_config_file():
    """Check if a valid configuration file is handled correctly."""
    try:
        config.reload_from_yaml_file("tests/config/valid_config.yaml")

        # Verify LLM providers
        assert "p1" in config.config.llm_providers.providers
        assert "p2" in config.config.llm_providers.providers
        assert config.config.llm_providers.providers["p1"].type == "bam"
        assert config.config.llm_providers.providers["p2"].type == "openai"

        # Verify OLS config
        assert config.ols_config.max_workers == 1
        assert config.ols_config.default_provider == "p1"
        assert config.ols_config.default_model == "m1"
        assert config.ols_config.user_data_collection is not None
        assert config.ols_config.user_data_collection.feedback_disabled is True
        assert config.ols_config.quota_handlers is not None

        # Verify MCP servers
        assert config.mcp_servers is not None
        # Only one server should remain (second one skipped due to missing secret file)
        assert len(config.mcp_servers.servers) == 1

        # First MCP server (the only one remaining)
        assert config.mcp_servers.servers[0].name == "foo"
        assert config.mcp_servers.servers[0].url == "http://localhost:8080"
        assert config.mcp_servers.servers[0].headers == {}
        assert config.mcp_servers.servers[0].timeout is None

        # Second MCP server ("bar") was skipped during validation
        # because its auth header references a non-existent file
    except Exception as e:
        print(traceback.format_exc())
        pytest.fail(f"loading valid configuration failed: {e}")


def test_valid_config_file_without_certificate_directory():
    """Check if a valid configuration file is handled correctly."""
    try:
        config.reload_from_yaml_file(
            "tests/config/valid_config_without_certificate_directory.yaml"
        )

        expected_config = Config(
            {
                "llm_providers": [
                    {
                        "name": "p1",
                        "type": "bam",
                        "url": "https://url1",
                        "credentials_path": "tests/config/secret/apitoken",
                        "models": [
                            {
                                "name": "m1",
                                "url": "https://murl1",
                                "credentials_path": "tests/config/secret/apitoken",
                                "context_window_size": 450,
                                "parameters": {"max_tokens_for_response": 100},
                            },
                            {
                                "name": "m2",
                                "url": "https://murl2",
                            },
                        ],
                    },
                    {
                        "name": "p2",
                        "type": "openai",
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
                        "indexes": [
                            {
                                "product_docs_index_path": "tests/config",
                                "product_docs_index_id": "product",
                            }
                        ],
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
                    "certificate_directory": constants.DEFAULT_CERTIFICATE_DIRECTORY,
                },
            }
        )
        assert config.config == expected_config
    except Exception as e:
        print(traceback.format_exc())
        pytest.fail(f"loading valid configuration failed: {e}")


def test_valid_config_file_with_postgres():
    """Check if a valid configuration file with Postgres conversation cache is handled correctly."""
    try:
        config.reload_from_yaml_file("tests/config/valid_config_postgres.yaml")

        expected_config = Config(
            {
                "llm_providers": [
                    {
                        "name": "p1",
                        "type": "bam",
                        "url": "https://url1",
                        "credentials_path": "tests/config/secret/apitoken",
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
                        "indexes": [
                            {
                                "product_docs_index_path": "tests/config",
                                "product_docs_index_id": "product",
                            }
                        ],
                    },
                    "conversation_cache": {
                        "type": "postgres",
                        "postgres": {
                            "host": "foobar.com",
                            "port": "1234",
                            "dbname": "test",
                            "user": "user",
                            "password_path": "tests/config/postgres_password.txt",
                            "ca_cert_path": "tests/config/postgres_cert.crt",
                            "ssl_mode": "require",
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
    except Exception as e:
        print(traceback.format_exc())
        pytest.fail(f"loading valid configuration failed: {e}")


def test_config_file_without_logging_config():
    """Check how a configuration file without logging config is correctly initialized."""
    # when logging configuration is not provided, default values will be used
    # it means the following call should not fail
    config.reload_from_yaml_file("tests/config/config_without_logging.yaml")

    # test if default values have been set
    logging_config = config.ols_config.logging_config
    assert logging_config is not None
    assert logging_config.app_log_level == logging.INFO
    assert logging_config.lib_log_level == logging.WARNING
    assert logging_config.uvicorn_log_level == logging.WARNING


def test_valid_config_without_query_filter():
    """Check if a valid configuration file without query filter creates empty regex filters."""
    config.reload_empty()
    assert config.query_redactor is not None
    assert config.query_redactor.regex_filters == []
    config.reload_from_yaml_file("tests/config/valid_config_without_query_filter.yaml")
    assert config.query_redactor.regex_filters == []


def test_valid_config_with_query_filter():
    """Check if a valid configuration file with query filter is handled correctly."""
    config.reload_empty()
    assert config.query_redactor is not None
    assert config.query_redactor.regex_filters == []
    config.reload_from_yaml_file("tests/config/valid_config_with_query_filter.yaml")
    assert config.query_redactor is not None
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


def test_valid_config_with_azure_openai():
    """Check if a valid configuration file with Azure OpenAI is handled correctly."""
    try:
        config.reload_from_yaml_file("tests/config/valid_config_with_azure_openai.yaml")

        expected_config = Config(
            {
                "llm_providers": [
                    {
                        "name": "p1",
                        "type": "azure_openai",
                        "url": "https://url1",
                        "credentials": "secret_key",
                        "deployment_name": "test",
                        "azure_openai_config": {
                            "url": "http://localhost:1234",
                            "deployment_name": "*deployment name*",
                        },
                        "models": [
                            {
                                "name": "m1",
                                "url": "https://murl1",
                            },
                        ],
                    },
                ],
                "ols_config": {
                    "conversation_cache": {
                        "type": "postgres",
                        "postgres": {
                            "host": "foobar.com",
                            "port": "1234",
                            "dbname": "test",
                            "user": "user",
                            "password_path": "tests/config/postgres_password.txt",
                            "ca_cert_path": "tests/config/postgres_cert.crt",
                            "ssl_mode": "require",
                        },
                    },
                    "default_provider": "p1",
                    "default_model": "m1",
                },
            }
        )
        assert config.config == expected_config
        provider_config = config.config.llm_providers.providers.get("p1")
        assert provider_config is not None
        assert provider_config.api_version == constants.DEFAULT_AZURE_API_VERSION
    except Exception as e:
        print(traceback.format_exc())
        pytest.fail(f"loading valid configuration failed: {e}")


def test_valid_config_with_azure_openai_credentials_path_only_in_provider_config():
    """Check if a valid configuration file with Azure OpenAI is handled correctly."""
    try:
        config.reload_from_yaml_file(
            "tests/config/valid_config_with_azure_openai_2.yaml"
        )

        expected_config = Config(
            {
                "llm_providers": [
                    {
                        "name": "p1",
                        "type": "azure_openai",
                        "url": "https://url1",
                        "api_key": "secret_key",
                        "deployment_name": "test",
                        "azure_openai_config": {
                            "url": "http://localhost:1234",
                            "deployment_name": "*deployment name*",
                            "api_key": "secret_key",
                            "credentials_path": "tests/config/secret/apitoken",
                        },
                        "models": [
                            {
                                "name": "m1",
                                "url": "https://murl1",
                            },
                        ],
                    },
                ],
                "ols_config": {
                    "conversation_cache": {
                        "type": "postgres",
                        "postgres": {
                            "host": "foobar.com",
                            "port": "1234",
                            "dbname": "test",
                            "user": "user",
                            "password_path": "tests/config/postgres_password.txt",
                            "ca_cert_path": "tests/config/postgres_cert.crt",
                            "ssl_mode": "require",
                        },
                    },
                    "default_provider": "p1",
                    "default_model": "m1",
                },
            }
        )
        assert config.config == expected_config
    except Exception as e:
        print(traceback.format_exc())
        pytest.fail(f"loading valid configuration failed: {e}")


def test_valid_config_with_azure_openai_api_version():
    """Check if a valid configuration file with Azure OpenAI is handled correctly."""
    try:
        config.reload_from_yaml_file(
            "tests/config/valid_config_with_azure_openai_api_version.yaml"
        )

        expected_config = Config(
            {
                "llm_providers": [
                    {
                        "name": "p1",
                        "type": "azure_openai",
                        "url": "https://url1",
                        "credentials": "secret_key",
                        "deployment_name": "test",
                        "api_version": "2023-12-31",
                        "azure_openai_config": {
                            "url": "http://localhost:1234",
                            "deployment_name": "*deployment name*",
                        },
                        "models": [
                            {
                                "name": "m1",
                                "url": "https://murl1",
                            },
                        ],
                    },
                ],
                "ols_config": {
                    "conversation_cache": {
                        "type": "postgres",
                        "postgres": {
                            "host": "foobar.com",
                            "port": "1234",
                            "dbname": "test",
                            "user": "user",
                            "password_path": "tests/config/postgres_password.txt",
                            "ca_cert_path": "tests/config/postgres_cert.crt",
                            "ssl_mode": "require",
                        },
                    },
                    "default_provider": "p1",
                    "default_model": "m1",
                },
            }
        )
        assert config.config == expected_config
        provider_config = config.config.llm_providers.providers.get("p1")
        assert provider_config is not None
        assert provider_config.api_version == "2024-12-31"
    except Exception as e:
        print(traceback.format_exc())
        pytest.fail(f"loading valid configuration failed: {e}")


def test_valid_config_with_bam():
    """Check if a valid configuration file with BAM is handled correctly."""
    try:
        config.reload_from_yaml_file("tests/config/valid_config_with_bam.yaml")

        expected_config = Config(
            {
                "llm_providers": [
                    {
                        "name": "p1",
                        "type": "bam",
                        "url": "https://url1",
                        "credentials_path": "tests/config/secret/apitoken",
                        "deployment_name": "test",
                        "bam_config": {
                            "url": "http://localhost:1234",
                            "credentials_path": "tests/config/secret/apitoken",
                        },
                        "models": [
                            {
                                "name": "m1",
                                "url": "https://murl1",
                            },
                        ],
                    },
                ],
                "ols_config": {
                    "conversation_cache": {
                        "type": "postgres",
                        "postgres": {
                            "host": "foobar.com",
                            "port": "1234",
                            "dbname": "test",
                            "user": "user",
                            "password_path": "tests/config/postgres_password.txt",
                            "ca_cert_path": "tests/config/postgres_cert.crt",
                            "ssl_mode": "require",
                        },
                    },
                    "default_provider": "p1",
                    "default_model": "m1",
                },
            }
        )
        assert config.config == expected_config
        provider_config = config.config.llm_providers.providers.get("p1")
        assert provider_config is not None
        assert provider_config.api_version is None
    except Exception as e:
        print(traceback.format_exc())
        pytest.fail(f"loading valid configuration failed: {e}")


def test_valid_config_with_bam_credentials_path_only_in_provider_config():
    """Check if a valid configuration file with BAM is handled correctly."""
    try:
        config.reload_from_yaml_file("tests/config/valid_config_with_bam_2.yaml")

        expected_config = Config(
            {
                "llm_providers": [
                    {
                        "name": "p1",
                        "type": "bam",
                        "url": "https://url1",
                        "deployment_name": "test",
                        "bam_config": {
                            "url": "http://localhost:1234",
                            "credentials_path": "tests/config/secret/apitoken",
                        },
                        "models": [
                            {
                                "name": "m1",
                                "url": "https://murl1",
                            },
                        ],
                    },
                ],
                "ols_config": {
                    "conversation_cache": {
                        "type": "postgres",
                        "postgres": {
                            "host": "foobar.com",
                            "port": "1234",
                            "dbname": "test",
                            "user": "user",
                            "password_path": "tests/config/postgres_password.txt",
                            "ca_cert_path": "tests/config/postgres_cert.crt",
                            "ssl_mode": "require",
                        },
                    },
                    "default_provider": "p1",
                    "default_model": "m1",
                },
            }
        )
        assert config.config == expected_config
    except Exception as e:
        print(traceback.format_exc())
        pytest.fail(f"loading valid configuration failed: {e}")


def test_valid_config_with_watsonx():
    """Check if a valid configuration file with Watsonx is handled correctly."""
    try:
        config.reload_from_yaml_file("tests/config/valid_config_with_watsonx.yaml")

        expected_config = Config(
            {
                "llm_providers": [
                    {
                        "name": "p1",
                        "type": "watsonx",
                        "url": "https://url1",
                        "credentials_path": "tests/config/secret/apitoken",
                        "deployment_name": "test",
                        "project_id": "project ID",
                        "watsonx_config": {
                            "url": "http://localhost:1234",
                            "project_id": "project ID",
                            "credentials_path": "tests/config/secret/apitoken",
                        },
                        "models": [
                            {
                                "name": "m1",
                                "url": "https://murl1",
                            },
                        ],
                    },
                ],
                "ols_config": {
                    "conversation_cache": {
                        "type": "postgres",
                        "postgres": {
                            "host": "foobar.com",
                            "port": "1234",
                            "dbname": "test",
                            "user": "user",
                            "password_path": "tests/config/postgres_password.txt",
                            "ca_cert_path": "tests/config/postgres_cert.crt",
                            "ssl_mode": "require",
                        },
                    },
                    "default_provider": "p1",
                    "default_model": "m1",
                },
            }
        )
        assert config.config == expected_config
    except Exception as e:
        print(traceback.format_exc())
        pytest.fail(f"loading valid configuration failed: {e}")


def test_valid_config_with_watsonx_credentials_path_only_in_provider_config():
    """Check if a valid configuration file with Watsonx is handled correctly."""
    try:
        config.reload_from_yaml_file("tests/config/valid_config_with_watsonx_2.yaml")

        expected_config = Config(
            {
                "llm_providers": [
                    {
                        "name": "p1",
                        "type": "watsonx",
                        "url": "https://url1",
                        "deployment_name": "test",
                        "project_id": "project ID",
                        "watsonx_config": {
                            "url": "http://localhost:1234",
                            "project_id": "project ID",
                            "credentials_path": "tests/config/secret/apitoken",
                        },
                        "models": [
                            {
                                "name": "m1",
                                "url": "https://murl1",
                            },
                        ],
                    },
                ],
                "ols_config": {
                    "conversation_cache": {
                        "type": "postgres",
                        "postgres": {
                            "host": "foobar.com",
                            "port": "1234",
                            "dbname": "test",
                            "user": "user",
                            "password_path": "tests/config/postgres_password.txt",
                            "ca_cert_path": "tests/config/postgres_cert.crt",
                            "ssl_mode": "require",
                        },
                    },
                    "default_provider": "p1",
                    "default_model": "m1",
                },
            }
        )
        assert config.config == expected_config
    except Exception as e:
        print(traceback.format_exc())
        pytest.fail(f"loading valid configuration failed: {e}")


def test_quota_limiters_property():
    """Check the quota handler property."""
    config.reload_empty()
    assert config is not None
    config.reload_from_yaml_file("tests/config/valid_config_without_query_filter.yaml")
    # force reinitialization
    assert config.quota_limiters is not None
