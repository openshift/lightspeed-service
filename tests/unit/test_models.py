"""Unit tests for data models."""
import logging

import pytest

from ols.app.models.config import (
    Config,
    ConversationCacheConfig,
    InvalidConfigurationError,
    LLMConfig,
    LoggerConfig,
    MemoryConfig,
    ModelConfig,
    OLSConfig,
    ProviderConfig,
    RedisConfig,
)
from ols.app.models.models import FeedbackRequest, LLMRequest


def test_feedback_request():
    """Test the FeedbackRequest model."""
    feedback_request = FeedbackRequest(
        conversation_id=123,
        feedback_object='{"rating": 5, "comment": "Great service!"}',
    )
    assert feedback_request.conversation_id == 123
    assert (
        feedback_request.feedback_object == '{"rating": 5, "comment": "Great service!"}'
    )


def test_llm_request():
    """Test the LLMRequest model."""
    llm_request = LLMRequest(query="Tell me about Kubernetes")
    assert llm_request.query == "Tell me about Kubernetes"
    assert llm_request.conversation_id is None
    assert llm_request.response is None

    llm_request = LLMRequest(
        query="Tell me about Kubernetes",
        conversation_id="abc",
        response="Kubernetes is a portable, extensible, open source platform ...",
    )
    assert llm_request.query == "Tell me about Kubernetes"
    assert llm_request.conversation_id == "abc"
    assert (
        llm_request.response
        == "Kubernetes is a portable, extensible, open source platform ..."
    )


def test_model_config():
    """Test the ModelConfig model."""
    model_config = ModelConfig(
        {
            "name": "test_name",
            "url": "test_url",
            "credential_path": "test_credential_path",
        }
    )

    assert model_config.name == "test_name"
    assert model_config.url == "test_url"
    assert model_config.credential_path == "test_credential_path"
    assert model_config.credentials is None

    model_config = ModelConfig()
    assert model_config.name is None
    assert model_config.url is None
    assert model_config.credential_path is None
    assert model_config.credentials is None


def test_provider_config():
    """Test the ProviderConfig model."""
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "url": "test_url",
            "credential_path": "test_credential_path",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "test_model_url",
                    "credential_path": "test_model_credential_path",
                }
            ],
        }
    )
    assert provider_config.name == "test_name"
    assert provider_config.url == "test_url"
    assert provider_config.credential_path == "test_credential_path"
    assert provider_config.credentials is None
    assert len(provider_config.models) == 1
    assert provider_config.models["test_model_name"].name == "test_model_name"
    assert provider_config.models["test_model_name"].url == "test_model_url"
    assert (
        provider_config.models["test_model_name"].credential_path
        == "test_model_credential_path"
    )
    assert provider_config.models["test_model_name"].credentials is None

    provider_config = ProviderConfig()
    assert provider_config.name is None
    assert provider_config.url is None
    assert provider_config.credential_path is None
    assert provider_config.credentials is None
    assert len(provider_config.models) == 0

    with pytest.raises(InvalidConfigurationError) as excinfo:
        ProviderConfig(
            {
                "name": "test_name",
                "url": "test_url",
                "credential_path": "test_credential_path",
                "models": [],
            }
        )
    assert "no models configured for provider" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        ProviderConfig(
            {
                "name": "test_name",
                "url": "test_url",
                "credential_path": "test_credential_path",
                "models": [
                    {
                        "url": "test_model_url",
                        "credential_path": "test_model_credential_path",
                    }
                ],
            }
        )
    assert "model name is missing" in str(excinfo.value)


def test_llm_config():
    """Test the LLMConfig model."""
    llm_config = LLMConfig(
        [
            {
                "name": "test_provider_name",
                "url": "test_provider_url",
                "credential_path": "test_provider_credential_path",
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credential_path": "test_model_credential_path",
                    }
                ],
            },
        ]
    )
    assert len(llm_config.providers) == 1
    assert llm_config.providers["test_provider_name"].name == "test_provider_name"
    assert llm_config.providers["test_provider_name"].url == "test_provider_url"
    assert (
        llm_config.providers["test_provider_name"].credential_path
        == "test_provider_credential_path"
    )
    assert llm_config.providers["test_provider_name"].credentials is None
    assert len(llm_config.providers["test_provider_name"].models) == 1
    assert (
        llm_config.providers["test_provider_name"].models["test_model_name"].name
        == "test_model_name"
    )
    assert (
        llm_config.providers["test_provider_name"].models["test_model_name"].url
        == "test_model_url"
    )
    assert (
        llm_config.providers["test_provider_name"]
        .models["test_model_name"]
        .credential_path
        == "test_model_credential_path"
    )
    assert (
        llm_config.providers["test_provider_name"].models["test_model_name"].credentials
        is None
    )

    llm_config = LLMConfig()
    assert len(llm_config.providers) == 0

    with pytest.raises(InvalidConfigurationError) as excinfo:
        LLMConfig(
            [
                {
                    "url": "test_provider_url",
                    "credential_path": "test_provider_credential_path",
                    "models": [],
                },
            ]
        )
    assert "provider name is missing" in str(excinfo.value)


def test_logger_config():
    """Test the LoggerConfig model."""
    logger_config = LoggerConfig(
        {
            "default_level": "INFO",
        }
    )
    assert logger_config.default_level == logging.INFO

    logger_config = LoggerConfig(
        {
            "default_level": logging.INFO,
        }
    )
    assert logger_config.default_level == "INFO"

    logger_config = LoggerConfig()
    assert logger_config.default_level is None


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
            "type": "in-memory",
            "in-memory": {
                "max_entries": 100,
            },
        }
    )
    assert conversation_cache_config.type == "in-memory"
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
    assert "redis config is missing" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        ConversationCacheConfig({"type": "in-memory"})
    assert "in-memory config is missing" in str(excinfo.value)


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
            "ols_enable_dev_ui": True,
            "conversation_cache": {
                "type": "in-memory",
                "in-memory": {
                    "max_entries": 100,
                },
            },
            "logger_config": {
                "default_level": "INFO",
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
    assert ols_config.enable_debug_ui is True
    assert ols_config.conversation_cache.type == "in-memory"
    assert ols_config.conversation_cache.memory.max_entries == 100
    assert ols_config.logger_config.default_level == logging.INFO


def test_config():
    """Test the Config model of the Global service configuration."""
    config = Config(
        {
            "llm_providers": [
                {
                    "name": "test_provider_name",
                    "url": "test_provider_url",
                    "credential_path": "test_provider_credential_path",
                    "models": [
                        {
                            "name": "test_model_name",
                            "url": "test_model_url",
                            "credential_path": "test_model_credential_path",
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
                "ols_enable_dev_ui": True,
                "conversation_cache": {
                    "type": "in-memory",
                    "in-memory": {
                        "max_entries": 100,
                    },
                },
                "logger_config": {
                    "default_level": "INFO",
                },
            },
        }
    )
    assert len(config.llm_config.providers) == 1
    assert (
        config.llm_config.providers["test_provider_name"].name == "test_provider_name"
    )
    assert config.llm_config.providers["test_provider_name"].url == "test_provider_url"
    assert (
        config.llm_config.providers["test_provider_name"].credential_path
        == "test_provider_credential_path"
    )
    assert config.llm_config.providers["test_provider_name"].credentials is None
    assert len(config.llm_config.providers["test_provider_name"].models) == 1
    assert (
        config.llm_config.providers["test_provider_name"].models["test_model_name"].name
        == "test_model_name"
    )
    assert (
        config.llm_config.providers["test_provider_name"].models["test_model_name"].url
        == "test_model_url"
    )
    assert (
        config.llm_config.providers["test_provider_name"]
        .models["test_model_name"]
        .credential_path
        == "test_model_credential_path"
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
    assert config.ols_config.enable_debug_ui is True
    assert config.ols_config.conversation_cache.type == "in-memory"
    assert config.ols_config.conversation_cache.memory.max_entries == 100
    assert config.ols_config.logger_config.default_level == logging.INFO

    with pytest.raises(InvalidConfigurationError) as excinfo:
        Config().validate()
    assert "no llm config found" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        Config({"llm_providers": []}).validate()
    assert "no llm providers found" in str(excinfo.value)

    # todo: activate test after merging https://github.com/openshift/lightspeed-service/pull/166
    # with pytest.raises(InvalidConfigurationError) as excinfo:
    #     Config(
    #         {
    #             "llm_providers": [
    #                 {
    #                     "name": "test_provider_name",
    #                     "url": "test_provider_url",
    #                     "credential_path": "test_provider_credential_path",
    #                     "models": [
    #                         {
    #                             "name": "test_model_name",
    #                             "url": "test_model_url",
    #                             "credential_path": "test_model_credential_path",
    #                         }
    #                     ],
    #                 }
    #             ],
    #         }
    #     ).validate()
    # assert "no ols config found" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        Config(
            {
                "llm_providers": [
                    {
                        "name": "test_provider_name",
                        "url": "test_provider_url",
                        "credential_path": "test_provider_credential_path",
                        "models": [
                            {
                                "name": "test_model_name",
                                "url": "test_model_url",
                                "credential_path": "test_model_credential_path",
                            }
                        ],
                    }
                ],
                "ols_config": {"default_provider": "test_default_provider"},
            }
        ).validate()
    assert "default model is not set" in str(excinfo.value)

    # todo: activate test after merging https://github.com/openshift/lightspeed-service/pull/166
    # and add test for the validations of conversation cache and logging config
    # with pytest.raises(InvalidConfigurationError) as excinfo:
    #     Config(
    #         {
    #             "llm_providers": [
    #                 {
    #                     "name": "test_provider_name",
    #                     "url": "test_provider_url",
    #                     "credential_path": "test_provider_credential_path",
    #                     "models": [
    #                         {
    #                             "name": "test_model_name",
    #                             "url": "test_model_url",
    #                             "credential_path": "test_model_credential_path",
    #                         }
    #                     ],
    #                 }
    #             ],
    #             "ols_config": {
    #                 "default_provider": "test_default_provider",
    #                 "default_model": "test_default_model",
    #             },
    #         }
    #     ).validate()
    # assert "classifier model is not set" in str(excinfo.value)
