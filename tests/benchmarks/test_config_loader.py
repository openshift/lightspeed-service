"""Benchmarks for the configuration loader."""

import io

import pytest

from ols import config
from ols.utils.checks import InvalidConfigurationError


def test_load_valid_config_file(benchmark):
    """Benchmark for configuration file loading when config file is valid."""
    benchmark(config.reload_from_yaml_file, "tests/config/valid_config.yaml")


def try_to_load_config_file(filename, exception_to_catch):
    """Try to load config file when it's known that exception is thrown."""
    with pytest.raises(exception_to_catch):
        config.reload_from_yaml_file(filename)


def test_load_invalid_config_file(benchmark):
    """Benchmark for configuration file loading when config file is invalid."""
    benchmark(
        try_to_load_config_file,
        "tests/config/invalid_config.yaml",
        InvalidConfigurationError,
    )


def test_load_non_existent_config(benchmark):
    """Benchmark for configuration file loading when config file does not exists."""
    benchmark(
        try_to_load_config_file, "tests/config/NOT_EXISTS.yaml", FileNotFoundError
    )


def read_empty_config_stream():
    """Check if a empty configuration stream is handled correctly."""
    with pytest.raises(Exception):
        # empty config won't be validated
        config._load_config_from_yaml_stream(io.StringIO(""))


def read_valid_config_stream():
    """Check if a valid configuration stream is handled correctly."""
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
      - name: m3
        url: 'https://murl3'
  - name: p2
    type: bam
    url: 'https://url2'
    models:
      - name: m1
        url: 'https://murl1'
      - name: m2
        url: 'https://murl2'
      - name: m3
        url: 'https://murl3'
ols_config:
  max_workers: 2
  conversation_cache:
    type: memory
    memory:
      max_entries: 1000
  logging_config:
    level: info
  default_provider: p1
  default_model: m1
dev_config:
  enable_dev_ui: true
  pyroscope_url: https://pyroscope.pyroscope.svc.cluster.local:4040
  disable_auth: false
  disable_tls: true
  llm_params:
    something: 5
"""))


def test_load_empty_config_stream(benchmark):
    """Benchmark for loading configuration from stream."""
    benchmark(read_empty_config_stream)


def test_load_valid_config_stream(benchmark):
    """Benchmark for loading configuration from stream."""
    benchmark(read_valid_config_stream)


def read_invalid_config_stream():
    """Check if a invalid configuration stream is handled correctly."""
    with pytest.raises(Exception):
        config._load_config_from_yaml_stream(io.StringIO("""
---
llm_providers:
  - name: p1
    type: bam-invalid
    url: 'http://url1'
    credentials_path: tests/config/secret/apitoken
    models:
      - name: m1
        url: 'http://murl1'
        credentials_path: tests/config/secret/apitoken
      - name: m2
        url: 'https://murl2'
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
dev_config:
  enable_dev_ui: true
  pyroscope_url: https://pyroscope.pyroscope.svc.cluster.local:4040
  disable_auth: false
  disable_tls: true
  llm_params:
    something: 5
"""))


def test_load_invalid_config_stream(benchmark):
    """Benchmark for loading invalid configuration from stream."""
    benchmark(read_invalid_config_stream)
