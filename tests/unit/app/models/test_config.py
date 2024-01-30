"""Unit tests for data models."""

import logging

import pytest

from ols.app.models.config import (
    Config,
    ConversationCacheConfig,
    InvalidConfigurationError,
    LLMProviders,
    LoggingConfig,
    MemoryConfig,
    ModelConfig,
    OLSConfig,
    ProviderConfig,
    RedisConfig,
)


def test_model_config():
    """Test the ModelConfig model."""
    model_config = ModelConfig(
        {
            "name": "test_name",
            "url": "test_url",
            "credentials_path": "tests/config/secret.txt",
        }
    )

    assert model_config.name == "test_name"
    assert model_config.url == "test_url"
    assert model_config.credentials == "secret_key"

    model_config = ModelConfig()
    assert model_config.name is None
    assert model_config.url is None
    assert model_config.credentials is None


def test_provider_config():
    """Test the ProviderConfig model."""
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "url": "test_url",
            "credentials_path": "tests/config/secret.txt",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "test_model_url",
                    "credentials_path": "tests/config/secret.txt",
                }
            ],
        }
    )
    assert provider_config.name == "test_name"
    assert provider_config.url == "test_url"
    assert provider_config.credentials == "secret_key"
    assert len(provider_config.models) == 1
    assert provider_config.models["test_model_name"].name == "test_model_name"
    assert provider_config.models["test_model_name"].url == "test_model_url"
    assert provider_config.models["test_model_name"].credentials == "secret_key"

    provider_config = ProviderConfig()
    assert provider_config.name is None
    assert provider_config.url is None
    assert provider_config.credentials is None
    assert len(provider_config.models) == 0

    with pytest.raises(InvalidConfigurationError) as excinfo:
        ProviderConfig(
            {
                "name": "test_name",
                "url": "test_url",
                "credentials_path": "tests/config/secret.txt",
                "models": [],
            }
        )
    assert "no models configured for provider" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        ProviderConfig(
            {
                "name": "test_name",
                "url": "test_url",
                "credentials_path": "tests/config/secret.txt",
                "models": [
                    {
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret.txt",
                    }
                ],
            }
        )
    assert "model name is missing" in str(excinfo.value)


def test_llm_providers():
    """Test the LLMProviders model."""
    llm_providers = LLMProviders(
        [
            {
                "name": "test_provider_name",
                "url": "test_provider_url",
                "credentials_path": "tests/config/secret.txt",
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret.txt",
                    }
                ],
            },
        ]
    )
    assert len(llm_providers.providers) == 1
    assert llm_providers.providers["test_provider_name"].name == "test_provider_name"
    assert llm_providers.providers["test_provider_name"].url == "test_provider_url"
    assert llm_providers.providers["test_provider_name"].credentials == "secret_key"
    assert len(llm_providers.providers["test_provider_name"].models) == 1
    assert (
        llm_providers.providers["test_provider_name"].models["test_model_name"].name
        == "test_model_name"
    )
    assert (
        llm_providers.providers["test_provider_name"].models["test_model_name"].url
        == "test_model_url"
    )
    assert (
        llm_providers.providers["test_provider_name"]
        .models["test_model_name"]
        .credentials
        == "secret_key"
    )

    llm_providers = LLMProviders()
    assert len(llm_providers.providers) == 0

    with pytest.raises(InvalidConfigurationError) as excinfo:
        LLMProviders(
            [
                {
                    "url": "test_provider_url",
                    "credentials_path": "tests/config/secret.txt",
                    "models": [],
                },
            ]
        )
    assert "provider name is missing" in str(excinfo.value)


class TestLoggingConfig:
    """Test the LoggingConfig model."""

    def test_valid_values(self):
        """Test valid values."""
        # test default values
        logging_config = LoggingConfig({})
        assert logging_config.app_log_level == logging.INFO
        assert logging_config.lib_log_level == logging.WARNING

        # test custom values
        logging_config = LoggingConfig(
            {
                "app_log_level": "debug",
                "lib_log_level": "debug",
            }
        )
        assert logging_config.app_log_level == logging.DEBUG
        assert logging_config.lib_log_level == logging.DEBUG

        logging_config = LoggingConfig()
        assert logging_config.app_log_level == logging.INFO

    def test_invalid_values(self):
        """Test invalid values."""
        # value is not string
        with pytest.raises(InvalidConfigurationError, match="invalid log level for 5"):
            LoggingConfig({"app_log_level": 5})

        # value is not valid log level
        with pytest.raises(
            InvalidConfigurationError,
            match="invalid log level for app_log_level: dingdong",
        ):
            LoggingConfig({"app_log_level": "dingdong"})


def test_redis_config():
    """Test the RedisConfig model."""
    redis_config = RedisConfig(
        {
            "host": "localhost",
            "port": 6379,
            "max_memory": "200mb",
            "max_memory_policy": "allkeys-lru",
        }
    )
    assert redis_config.host == "localhost"
    assert redis_config.port == 6379
    assert redis_config.max_memory == "200mb"
    assert redis_config.max_memory_policy == "allkeys-lru"

    redis_config = RedisConfig()
    assert redis_config.host is None
    assert redis_config.port is None
    assert redis_config.max_memory is None
    assert redis_config.max_memory_policy is None


def test_memory_cache_config():
    """Test the MemoryCacheConfig model."""
    memory_cache_config = MemoryConfig(
        {
            "max_entries": 100,
        }
    )
    assert memory_cache_config.max_entries == 100

    memory_cache_config = MemoryConfig()
    assert memory_cache_config.max_entries is None


def test_conversation_cache_config():
    """Test the ConversationCacheConfig model."""
    conversation_cache_config = ConversationCacheConfig(
        {
            "type": "memory",
            "memory": {
                "max_entries": 100,
            },
        }
    )
    assert conversation_cache_config.type == "memory"
    assert conversation_cache_config.memory.max_entries == 100

    conversation_cache_config = ConversationCacheConfig(
        {
            "type": "redis",
            "redis": {
                "host": "localhost",
                "port": 6379,
                "max_memory": "200mb",
                "max_memory_policy": "allkeys-lru",
            },
        }
    )
    assert conversation_cache_config.type == "redis"
    assert conversation_cache_config.redis.host == "localhost"
    assert conversation_cache_config.redis.port == 6379
    assert conversation_cache_config.redis.max_memory == "200mb"
    assert conversation_cache_config.redis.max_memory_policy == "allkeys-lru"

    conversation_cache_config = ConversationCacheConfig()
    assert conversation_cache_config.type is None
    assert conversation_cache_config.redis is None
    assert conversation_cache_config.memory is None

    with pytest.raises(InvalidConfigurationError) as excinfo:
        ConversationCacheConfig({"type": "redis"})
    assert "redis configuration is missing" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        ConversationCacheConfig({"type": "memory"})
    assert "memory configuration is missing" in str(excinfo.value)


def test_ols_config():
    """Test the OLSConfig model."""
    ols_config = OLSConfig(
        {
            "default_provider": "test_default_provider",
            "default_model": "test_default_model",
            "classifier_provider": "test_classifer_provider",
            "classifier_model": "test_classifier_model",
            "summarizer_provider": "test_summarizer_provider",
            "summarizer_model": "test_summarizer_model",
            "validator_provider": "test_validator_provider",
            "validator_model": "test_validator_model",
            "yaml_provider": "test_yaml_provider",
            "yaml_model": "test_yaml_model",
            "conversation_cache": {
                "type": "memory",
                "memory": {
                    "max_entries": 100,
                },
            },
            "logging_config": {
                "logging_level": "INFO",
            },
        }
    )
    assert ols_config.default_provider == "test_default_provider"
    assert ols_config.default_model == "test_default_model"
    assert ols_config.classifier_provider == "test_classifer_provider"
    assert ols_config.classifier_model == "test_classifier_model"
    assert ols_config.summarizer_provider == "test_summarizer_provider"
    assert ols_config.summarizer_model == "test_summarizer_model"
    assert ols_config.validator_provider == "test_validator_provider"
    assert ols_config.validator_model == "test_validator_model"
    assert ols_config.yaml_provider == "test_yaml_provider"
    assert ols_config.yaml_model == "test_yaml_model"
    assert ols_config.conversation_cache.type == "memory"
    assert ols_config.conversation_cache.memory.max_entries == 100
    assert ols_config.logging_config.app_log_level == logging.INFO


def test_config():
    """Test the Config model of the Global service configuration."""
    config = Config(
        {
            "llm_providers": [
                {
                    "name": "test_provider_name",
                    "url": "test_provider_url",
                    "credentials_path": "tests/config/secret.txt",
                    "models": [
                        {
                            "name": "test_model_name",
                            "url": "test_model_url",
                            "credentials_path": "tests/config/secret.txt",
                        }
                    ],
                },
            ],
            "ols_config": {
                "default_provider": "test_default_provider",
                "default_model": "test_default_model",
                "classifier_provider": "test_classifer_provider",
                "classifier_model": "test_classifier_model",
                "summarizer_provider": "test_summarizer_provider",
                "summarizer_model": "test_summarizer_model",
                "validator_provider": "test_validator_provider",
                "validator_model": "test_validator_model",
                "yaml_provider": "test_yaml_provider",
                "yaml_model": "test_yaml_model",
                "conversation_cache": {
                    "type": "memory",
                    "memory": {
                        "max_entries": 100,
                    },
                },
                "logging_config": {
                    "app_log_level": "error",
                },
            },
        }
    )
    assert len(config.llm_providers.providers) == 1
    assert (
        config.llm_providers.providers["test_provider_name"].name
        == "test_provider_name"
    )
    assert (
        config.llm_providers.providers["test_provider_name"].url == "test_provider_url"
    )
    assert (
        config.llm_providers.providers["test_provider_name"].credentials == "secret_key"
    )
    assert len(config.llm_providers.providers["test_provider_name"].models) == 1
    assert (
        config.llm_providers.providers["test_provider_name"]
        .models["test_model_name"]
        .name
        == "test_model_name"
    )
    assert (
        config.llm_providers.providers["test_provider_name"]
        .models["test_model_name"]
        .url
        == "test_model_url"
    )
    assert (
        config.llm_providers.providers["test_provider_name"]
        .models["test_model_name"]
        .credentials
        == "secret_key"
    )
    assert config.ols_config.default_provider == "test_default_provider"
    assert config.ols_config.default_model == "test_default_model"
    assert config.ols_config.classifier_provider == "test_classifer_provider"
    assert config.ols_config.classifier_model == "test_classifier_model"
    assert config.ols_config.summarizer_provider == "test_summarizer_provider"
    assert config.ols_config.summarizer_model == "test_summarizer_model"
    assert config.ols_config.validator_provider == "test_validator_provider"
    assert config.ols_config.validator_model == "test_validator_model"
    assert config.ols_config.yaml_provider == "test_yaml_provider"
    assert config.ols_config.yaml_model == "test_yaml_model"
    assert config.ols_config.conversation_cache.type == "memory"
    assert config.ols_config.conversation_cache.memory.max_entries == 100
    assert config.ols_config.logging_config.app_log_level == logging.ERROR

    with pytest.raises(InvalidConfigurationError) as excinfo:
        Config().validate_yaml()
    assert "no LLM providers config section found" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        Config({"llm_providers": []}).validate_yaml()
    assert "no OLS config section found" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        Config(
            {
                "llm_providers": [
                    {
                        "name": "test_provider_name",
                        "url": "http://test_provider_url",
                        "credentials_path": "tests/config/secret.txt",
                        "models": [
                            {
                                "name": "test_model_name",
                                "url": "http://test_model_url",
                                "credentials_path": "tests/config/secret.txt",
                            }
                        ],
                    }
                ],
            }
        ).validate_yaml()
    assert "no OLS config section found" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        Config(
            {
                "llm_providers": [
                    {
                        "name": "test_provider_name",
                        "url": "http://test_provider_url",
                        "credentials_path": "tests/config/secret.txt",
                        "models": [
                            {
                                "name": "test_model_name",
                                "url": "http://test_model_url",
                                "credentials_path": "tests/config/secret.txt",
                            }
                        ],
                    }
                ],
                "ols_config": {"default_provider": "test_default_provider"},
            }
        ).validate_yaml()
    assert "default_provider is specified, but default_model is missing" in str(
        excinfo.value
    )
