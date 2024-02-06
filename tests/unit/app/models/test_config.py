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
    RedisCredentials,
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


def test_model_config_equality():
    """Test the ModelConfig equality check."""
    model_config_1 = ModelConfig()
    model_config_2 = ModelConfig()

    # compare the same model configs
    assert model_config_1 == model_config_2

    # compare different model configs
    model_config_2.name = "some non-default name"
    assert model_config_1 != model_config_2

    # compare with value of different type
    other_value = "foo"
    assert model_config_1 != other_value


def test_model_config_validation_proper_config():
    """Test the ModelConfig model validation."""
    model_config = ModelConfig(
        {
            "name": "test_name",
            "url": "http://test.url",
            "credentials_path": "tests/config/secret.txt",
        }
    )
    # validation should not fail
    model_config.validate_yaml()


def test_model_config_validation_no_credentials_path():
    """Test the ModelConfig model validation when path to credentials is not provided."""
    model_config = ModelConfig(
        {
            "name": "test_name",
            "url": "http://test.url",
            "credentials_path": None,
        }
    )
    # validation should not fail
    model_config.validate_yaml()
    assert model_config.credentials is None


def test_model_config_validation_empty_model():
    """Test the ModelConfig model validation when model is empty."""
    model_config = ModelConfig()

    # validation should fail
    with pytest.raises(InvalidConfigurationError, match="model name is missing"):
        model_config.validate_yaml()


def test_model_config_validation_missing_name():
    """Test the ModelConfig model validation when model name is missing."""
    model_config = ModelConfig(
        {
            "name": None,
            "url": "http://test.url",
            "credentials_path": "tests/config/secret.txt",
        }
    )

    # validation should fail
    with pytest.raises(InvalidConfigurationError, match="model name is missing"):
        model_config.validate_yaml()


def test_model_config_validation_improper_url():
    """Test the ModelConfig model validation when URL is incorrect."""
    model_config = ModelConfig(
        {
            "name": "test_name",
            "url": "httpXXX://test.url",
            "credentials_path": "tests/config/secret.txt",
        }
    )

    # validation should fail
    with pytest.raises(InvalidConfigurationError, match="model URL is invalid"):
        model_config.validate_yaml()


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


def test_provider_config_equality():
    """Test the ProviderConfig equality check."""
    provider_config_1 = ProviderConfig()
    provider_config_2 = ProviderConfig()

    # compare the same provider configs
    assert provider_config_1 == provider_config_2

    # compare different model configs
    provider_config_2.name = "some non-default name"
    assert provider_config_1 != provider_config_2

    # compare with value of different type
    other_value = "foo"
    assert provider_config_1 != other_value


def test_provider_config_validation_proper_config():
    """Test the ProviderConfig model validation."""
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "url": "http://test.url",
            "credentials_path": "tests/config/secret.txt",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.model.url",
                    "credentials_path": "tests/config/secret.txt",
                }
            ],
        }
    )

    provider_config.validate_yaml()


def test_provider_config_validation_improper_url():
    """Test the ProviderConfig model validation for improper URL."""
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "url": "httpXXX://test.url",
            "credentials_path": "tests/config/secret.txt",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.model.url",
                    "credentials_path": "tests/config/secret.txt",
                }
            ],
        }
    )

    with pytest.raises(InvalidConfigurationError, match="provider URL is invalid"):
        provider_config.validate_yaml()


def test_provider_config_validation_missing_name():
    """Test the ProviderConfig model validation for missing name."""
    provider_config = ProviderConfig(
        {
            "url": "httpXXX://test.url",
            "credentials_path": "tests/config/secret.txt",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.model.url",
                    "credentials_path": "tests/config/secret.txt",
                }
            ],
        }
    )

    with pytest.raises(InvalidConfigurationError, match="provider name is missing"):
        provider_config.validate_yaml()


def test_provider_config_validation_no_credentials_path():
    """Test the ProviderConfig model validation when path to credentials is not provided."""
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "url": "http://test.url",
            "credentials_path": None,
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.model.url",
                    "credentials_path": "tests/config/secret.txt",
                }
            ],
        }
    )

    provider_config.validate_yaml()
    assert provider_config.credentials is None


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


def test_llm_providers_equality():
    """Test the LLMProviders equality check."""
    provider_config_1 = LLMProviders()
    provider_config_2 = LLMProviders()

    # compare same providers
    assert provider_config_1 == provider_config_2

    # compare different providers
    provider_config_2.providers = [ProviderConfig()]
    assert provider_config_1 != provider_config_2

    # compare with value of different type
    other_value = "foo"
    assert provider_config_1 != other_value


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


def test_redis_config_with_credentials():
    """Test the RedisConfig model."""
    redis_config = RedisConfig(
        {
            "host": "localhost",
            "port": 6379,
            "max_memory": "200mb",
            "max_memory_policy": "allkeys-lru",
            "credentials": {
                "username_path": "tests/config/redis_username.txt",
                "password_path": "tests/config/redis_password.txt",
            },
        }
    )
    assert redis_config.credentials.username == "redis_username"
    assert redis_config.credentials.password == "redis_password"  # noqa S105


def test_redis_credentials_validation_empty_setup():
    """Test the validation of empty RedisCredentials."""
    redis_credentials = RedisCredentials()
    redis_credentials.validate_yaml()


def test_redis_credentials_validation_correct_setup():
    """Test the validation of correct RedisCredentials."""
    redis_credentials = RedisCredentials(
        {
            "username_path": "tests/config/redis_username.txt",
            "password_path": "tests/config/redis_password.txt",
        }
    )
    redis_credentials.validate_yaml()


def test_redis_credentials_validation_missing_password_path():
    """Test the validation of incorrect RedisCredentials."""
    redis_credentials = RedisCredentials(
        {
            "username_path": "tests/config/redis_username.txt",
            "password_path": None,
        }
    )
    with pytest.raises(
        InvalidConfigurationError,
        match="for Redis, if a username is specified, a password also needs to be specified",
    ):
        redis_credentials.validate_yaml()


def test_redis_credentials_validation_missing_user_path():
    """Test the validation of missing username in RedisCredentials."""
    redis_credentials = RedisCredentials(
        {
            "username_path": None,
            "password_path": "tests/config/redis_password.txt",
        }
    )
    redis_credentials.validate_yaml()


def test_redis_credentials_validation_wrong_paths():
    """Test the validation of incorrect paths to credentials."""
    with pytest.raises(Exception):
        RedisCredentials(
            {
                "username_path": "tests/config/redis_username.txt",
                "password_path": "/dev/null/foobar",
            }
        )
    with pytest.raises(Exception):
        RedisCredentials(
            {
                "username_path": "/dev/null/foobar",
                "password_path": "tests/config/redis_password.txt",
            }
        )


def test_redis_credentials_equality():
    """Test the RedisCredentials equality check."""
    redis_credentials_1 = RedisCredentials()
    redis_credentials_2 = RedisCredentials()

    # compare the same Redis credentialss
    assert redis_credentials_1 == redis_credentials_2

    # compare different Redis credentialss
    redis_credentials_2.username = "this is me"
    assert redis_credentials_1 != redis_credentials_2

    # compare with value of different type
    other_value = "foo"
    assert redis_credentials_1 != other_value


def test_redis_config_equality():
    """Test the RedisConfig equality check."""
    redis_config_1 = RedisConfig()
    redis_config_2 = RedisConfig()

    # compare the same Redis configs
    assert redis_config_1 == redis_config_2

    # compare different Redis configs
    redis_config_2.host = "12.34.56.78"
    assert redis_config_1 != redis_config_2

    # compare with value of different type
    other_value = "foo"
    assert redis_config_1 != other_value


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


def test_memory_cache_config_improper_entries():
    """Test the MemoryCacheConfig model if improper max_entries is used."""
    with pytest.raises(
        InvalidConfigurationError,
        match="invalid max_entries for memory conversation cache",
    ):
        MemoryConfig(
            {
                "max_entries": -100,
            }
        )


def test_memory_config_equality():
    """Test the MemoryConfig equality check."""
    memory_config_1 = MemoryConfig()
    memory_config_2 = MemoryConfig()

    # compare the same memory configs
    assert memory_config_1 == memory_config_2

    # compare different memory configs
    memory_config_2.max_entries = 123456
    assert memory_config_1 != memory_config_2

    # compare with value of different type
    other_value = "foo"
    assert memory_config_1 != other_value


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


def test_conversation_cache_config_validation():
    """Test the ConversationCacheConfig validation."""
    conversation_cache_config = ConversationCacheConfig()

    # not specified cache type case
    conversation_cache_config.type = None
    with pytest.raises(
        InvalidConfigurationError, match="missing conversation cache type"
    ):
        conversation_cache_config.validate_yaml()

    # unknown cache type case
    conversation_cache_config.type = "unknown"
    with pytest.raises(
        InvalidConfigurationError, match="unknown conversation cache type: unknown"
    ):
        conversation_cache_config.validate_yaml()


def test_ols_config():
    """Test the OLSConfig model."""
    ols_config = OLSConfig(
        {
            "default_provider": "test_default_provider",
            "default_model": "test_default_model",
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
    assert config.ols_config.conversation_cache.type == "memory"
    assert config.ols_config.conversation_cache.memory.max_entries == 100
    assert config.ols_config.logging_config.app_log_level == logging.ERROR


def test_config_no_llm_providers():
    """Check if empty config is rejected as expected."""
    with pytest.raises(
        InvalidConfigurationError, match="no LLM providers config section found"
    ):
        Config().validate_yaml()


def test_config_empty_llm_providers():
    """Check if empty list of providers is rejected as expected."""
    with pytest.raises(InvalidConfigurationError, match="no OLS config section found"):
        Config({"llm_providers": []}).validate_yaml()


def test_config_without_ols_section():
    """Test the Config model of the Global service configuration with missing OLS section."""
    with pytest.raises(InvalidConfigurationError, match="no OLS config section found"):
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


def test_config_improper_missing_model():
    """Test the Config model of the Global service configuration when model is missing."""
    with pytest.raises(
        InvalidConfigurationError,
        match="default_provider is specified, but default_model is missing",
    ):
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
                "ols_config": {
                    "conversation_cache": {
                        "type": "memory",
                        "memory": {
                            "max_entries": 1000,
                        },
                    },
                    "default_provider": "test_default_provider",
                },
            }
        ).validate_yaml()


def test_config_improper_provider():
    """Test the Config model of the Global service configuration when improper provider is set."""
    with pytest.raises(
        InvalidConfigurationError,
        match="default_provider specifies an unknown provider test_default_provider",
    ):
        Config(
            {
                "llm_providers": [],
                "ols_config": {
                    "default_provider": "test_default_provider",
                    "default_model": "test_default_model",
                    "classifier_provider": "test_classifer_provider",
                    "classifier_model": "test_classifier_model",
                    "summarizer_provider": "test_summarizer_provider",
                    "summarizer_model": "test_summarizer_model",
                    "validator_provider": "test_validator_provider",
                    "validator_model": "test_validator_model",
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
        ).validate_yaml()


def test_config_improper_model():
    """Test the Config model of the Global service configuration when improper model is set."""
    with pytest.raises(
        InvalidConfigurationError,
        match="default_model specifies an unknown model test_default_model",
    ):
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
                "ols_config": {
                    "default_provider": "test_provider_name",
                    "default_model": "test_default_model",
                    "classifier_provider": "test_classifer_provider",
                    "classifier_model": "test_classifier_model",
                    "summarizer_provider": "test_summarizer_provider",
                    "summarizer_model": "test_summarizer_model",
                    "validator_provider": "test_validator_provider",
                    "validator_model": "test_validator_model",
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
        ).validate_yaml()
