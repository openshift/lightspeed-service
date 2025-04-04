"""Unit tests for data models."""

import copy
import logging

import pytest
from pydantic import ValidationError

import ols.utils.tls as tls
from ols import constants
from ols.app.models.config import (
    AuthenticationConfig,
    Config,
    ConversationCacheConfig,
    DevConfig,
    InMemoryCacheConfig,
    LLMProviders,
    LoggingConfig,
    ModelConfig,
    ModelParameters,
    OLSConfig,
    PostgresConfig,
    ProviderConfig,
    QueryFilter,
    QuotaHandlersConfig,
    RedisConfig,
    ReferenceContent,
    ReferenceContentIndex,
    TLSConfig,
    TLSSecurityProfile,
    UserDataCollection,
    UserDataCollectorConfig,
)
from ols.utils.checks import InvalidConfigurationError


def test_model_parameters():
    """Test the ModelParameters model."""
    default_params = ModelParameters()
    assert (
        default_params.max_tokens_for_response
        == constants.DEFAULT_MAX_TOKENS_FOR_RESPONSE
    )

    parameters = ModelParameters(max_tokens_for_response=10, unknown_param="hello")

    assert parameters.max_tokens_for_response == 10
    assert not hasattr(parameters, "unknown_param")

    # max_tokens_for_response needs to be positive integer
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        ModelParameters(max_tokens_for_response=-1)


def test_model_config():
    """Test the ModelConfig model."""
    model_config = ModelConfig(
        name="test_name",
        url="http://test_url/",
        credentials_path="tests/config/secret/apitoken",
        options={
            "foo": 1,
            "bar": 2,
        },
    )

    assert model_config.name == "test_name"
    assert str(model_config.url) == "http://test_url/"
    assert model_config.credentials == "secret_key"
    assert model_config.options == {
        "foo": 1,
        "bar": 2,
    }

    model_config = ModelConfig(name="a")
    assert model_config.name == "a"
    assert model_config.url is None
    assert model_config.credentials is None
    assert model_config.options is None


def test_model_config_path_to_secret_directory():
    """Test the ModelConfig model."""
    model_config = ModelConfig(
        name="test_name",
        url="http://test_url/",
        credentials_path="tests/config/secret",
        options={
            "foo": 1,
            "bar": 2,
        },
    )

    assert model_config.credentials == "secret_key"


def test_model_config_equality():
    """Test the ModelConfig equality check."""
    model_config_1 = ModelConfig(name="a")
    model_config_2 = ModelConfig(name="a")

    # compare the same model configs
    assert model_config_1 == model_config_2

    # compare different model configs
    model_config_2.name = "some non-default name"
    assert model_config_1 != model_config_2

    # compare with value of different type
    other_value = "foo"
    assert model_config_1 != other_value


def test_model_config_no_options():
    """Test the ModelConfig without options."""
    ModelConfig(
        name="test_name",
        url="http://test.url",
        credentials_path="tests/config/secret/apitoken",
    )


def test_model_config_validation_no_credentials_path():
    """Test the ModelConfig model validation when path to credentials is not provided."""
    model_config = ModelConfig(
        name="test_name",
        url="http://test.url",
        credentials_path=None,
    )
    assert model_config.credentials is None


def test_model_config_validation_empty_model():
    """Test the ModelConfig model validation when model is empty."""
    with pytest.raises(InvalidConfigurationError, match="model name is missing"):
        ModelConfig()


def test_model_config_wrong_options():
    """Test the ModelConfig model validation."""
    with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
        ModelConfig(
            name="test_name",
            url="http://test.url",
            credentials_path="tests/config/secret/apitoken",
            options="not-dictionary",
        )


def test_model_config_wrong_option_key():
    """Test the ModelConfig model validation."""
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        ModelConfig(
            name="test_name",
            url="http://test.url",
            credentials_path="tests/config/secret/apitoken",
            options={
                42: "answer",
            },
        )


def test_model_config_validation_improper_url():
    """Test the ModelConfig model validation when URL is incorrect."""
    with pytest.raises(ValidationError, match="URL scheme should be 'http' or 'https'"):
        ModelConfig(
            name="test_name",
            url="httpXXX://test.url",
            credentials_path="tests/config/secret/apitoken",
        )


def test_model_config_higher_response_token():
    """Test the model config with response token >= context window."""
    with pytest.raises(
        InvalidConfigurationError,
        match="Context window size 2, should be greater than max_tokens_for_response 2",
    ):
        ModelConfig(
            name="test_model_name",
            context_window_size=2,
            parameters=ModelParameters(max_tokens_for_response=2),
        )


def test_provider_config():
    """Test the ProviderConfig model."""
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": "bam",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "test_project_id",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )
    assert provider_config.name == "test_name"
    assert provider_config.type == "bam"
    assert provider_config.url == "test_url"
    assert provider_config.credentials == "secret_key"
    assert provider_config.project_id == "test_project_id"
    assert len(provider_config.models) == 1
    assert provider_config.models["test_model_name"].name == "test_model_name"
    assert str(provider_config.models["test_model_name"].url) == "http://test.url/"
    assert provider_config.models["test_model_name"].credentials == "secret_key"
    assert (
        provider_config.models["test_model_name"].context_window_size
        == constants.DEFAULT_CONTEXT_WINDOW_SIZE
    )
    assert (
        provider_config.models["test_model_name"].parameters.max_tokens_for_response
        == constants.DEFAULT_MAX_TOKENS_FOR_RESPONSE
    )
    assert provider_config.openai_config is None
    assert provider_config.azure_config is None
    assert provider_config.watsonx_config is None
    assert provider_config.bam_config is None

    provider_config = ProviderConfig()
    assert provider_config.name is None
    assert provider_config.url is None
    assert provider_config.credentials is None
    assert provider_config.project_id is None
    assert len(provider_config.models) == 0

    assert provider_config.openai_config is None
    assert provider_config.azure_config is None
    assert provider_config.watsonx_config is None
    assert provider_config.bam_config is None
    assert provider_config.tls_security_profile is None

    with pytest.raises(InvalidConfigurationError) as excinfo:
        ProviderConfig(
            {
                "name": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "models": [],
            }
        )
    assert "no models configured for provider" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        ProviderConfig(
            {
                "name": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "models": [
                    {
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )
    assert "model name is missing" in str(excinfo.value)


def test_provider_config_improper_path_to_secret():
    """Test that exception is thrown when path to secret is wrong."""
    with pytest.raises(FileNotFoundError):
        ProviderConfig(
            {
                "name": "bam",
                "url": "test_url",
                "credentials_path": "foo",
                "models": [
                    {
                        "name": "WatsonX",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )

    # now let's ignore LLM secrets-related errors
    ProviderConfig(
        {
            "name": "bam",
            "url": "test_url",
            "credentials_path": "foo",
            "models": [
                {
                    "name": "WatsonX",
                    "url": "http://test.url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        },
        ignore_llm_secrets=True,
    )


def test_provider_config_with_tls_security_profile():
    """Test the ProviderConfig model."""
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": "bam",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "test_project_id",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
            "tlsSecurityProfile": {
                "type": "Custom",
                "minTLSVersion": "VersionTLS13",
                "ciphers": [
                    "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
                    "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
                ],
            },
        }
    )
    assert provider_config.tls_security_profile is not None
    assert provider_config.tls_security_profile.profile_type == "Custom"
    assert provider_config.tls_security_profile.min_tls_version == "VersionTLS13"
    assert (
        "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
        in provider_config.tls_security_profile.ciphers
    )
    assert (
        "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"
        in provider_config.tls_security_profile.ciphers
    )


def test_that_url_is_required_provider_parameter():
    """Test that provider-specific URL is required attribute."""
    # provider type is set to "azure_openai"
    with pytest.raises(ValidationError, match="url"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "azure_openai",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "deployment_name": "deploment-name",
                "azure_openai_config": {
                    "tenant_id": "tenant-ID",
                    "client_id": "client-ID",
                    "client_secret_path": "tests/config/secret/apitoken",
                    "credentials_path": "tests/config/secret/apitoken",
                    "deployment_name": "deployment-name",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )

    # provider type is set to "openai"
    with pytest.raises(ValidationError, match="url"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "openai",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "openai_config": {
                    "credentials_path": "tests/config/secret/apitoken",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )

    # provider type is set to "bam"
    with pytest.raises(ValidationError, match="url"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "bam_config": {
                    "credentials_path": "tests/config/secret/apitoken",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )

    # provider type is set to "watsonx"
    with pytest.raises(ValidationError, match="url"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "watsonx",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "watsonx_config": {
                    "credentials_path": "tests/config/secret/apitoken",
                    "project_id": "*project id*",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


def test_that_credentials_is_required_provider_parameter():
    """Test that provider-specific credentials is required attribute for any provider but Azure."""
    # provider type is set to "openai"
    with pytest.raises(ValidationError, match="credentials"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "openai",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "openai_config": {
                    "url": "http://localhost",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )

    # provider type is set to "bam"
    with pytest.raises(ValidationError, match="credentials"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "bam_config": {
                    "url": "http://localhost",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )

    # provider type is set to "watsonx"
    with pytest.raises(ValidationError, match="credentials"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "watsonx",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "watsonx_config": {
                    "project_id": "*project id*",
                    "url": "http://localhost",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


def test_provider_config_azure_openai_specific():
    """Test if Azure OpenAI-specific config is loaded and validated."""
    # provider type is set to "azure_openai" and Azure OpenAI-specific configuration is there
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": "azure_openai",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "deployment_name": "deploment-name",
            "azure_openai_config": {
                "url": "http://localhost",
                "credentials_path": "tests/config/secret_azure_tenant_id_client_id_client_secret",
                "deployment_name": "deployment-name",
            },
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )
    # Default azure api version
    assert provider_config.api_version == constants.DEFAULT_AZURE_API_VERSION

    # Azure OpenAI-specific configuration must be present
    assert provider_config.azure_config is not None
    assert str(provider_config.azure_config.url) == "http://localhost/"
    assert (
        provider_config.azure_config.tenant_id == "00000000-0000-0000-0000-000000000001"
    )
    assert (
        provider_config.azure_config.client_id == "00000000-0000-0000-0000-000000000002"
    )
    assert provider_config.azure_config.deployment_name == "deployment-name"
    assert provider_config.azure_config.client_secret == "client secret"  # noqa: S105
    assert provider_config.azure_config.api_key is None

    # configuration for other providers must not be set
    assert provider_config.openai_config is None
    assert provider_config.rhoai_vllm_config is None
    assert provider_config.rhelai_vllm_config is None
    assert provider_config.watsonx_config is None
    assert provider_config.bam_config is None


def test_provider_config_apitoken_only():
    """Test if Azure OpenAI-specific config is loaded and validated."""
    # provider type is set to "azure_openai" and Azure OpenAI-specific configuration is there
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": "azure_openai",
            "url": "test_url",
            "api_version": "2024-02-15",
            "azure_openai_config": {
                "url": "http://localhost",
                "credentials_path": "tests/config/secret/apitoken",
                "deployment_name": "deployment-name",
            },
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.url/",
                }
            ],
        }
    )
    # Azure version is set from config
    assert provider_config.api_version == "2024-02-15"

    # Azure OpenAI-specific configuration must be present
    assert provider_config.azure_config is not None
    assert str(provider_config.azure_config.url) == "http://localhost/"
    assert provider_config.azure_config.tenant_id is None
    assert provider_config.azure_config.client_id is None
    assert provider_config.azure_config.client_secret is None

    assert provider_config.azure_config.deployment_name == "deployment-name"
    assert provider_config.azure_config.api_key is not None

    # configuration for other providers must not be set
    assert provider_config.openai_config is None
    assert provider_config.rhoai_vllm_config is None
    assert provider_config.rhelai_vllm_config is None
    assert provider_config.watsonx_config is None
    assert provider_config.bam_config is None


def test_provider_config_azure_openai_unknown_parameters():
    """Test if unknown Azure OpenAI parameters are detected."""
    # provider type is set to "azure_openai" and Azure OpenAI-specific configuration is there
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "azure_openai",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "deployment_name": "deploment-name",
                "azure_openai_config": {
                    "unknown_parameter": "unknown value",
                    "url": "http://localhost",
                    "tenant_id": "tenant-ID",
                    "client_id": "client-ID",
                    "client_secret_path": "tests/config/secret/apitoken",
                    "credentials_path": "tests/config/secret/apitoken",
                    "deployment_name": "deployment-name",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


def test_provider_config_openai_specific():
    """Test if OpenAI-specific config is loaded and validated."""
    # provider type is set to "openai" and OpenAI-specific configuration is there
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": "openai",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "test_project_id",
            "openai_config": {
                "url": "http://localhost",
                "credentials_path": "tests/config/secret/apitoken",
            },
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )
    # OpenAI-specific configuration must be present
    assert provider_config.openai_config is not None
    assert str(provider_config.openai_config.url) == "http://localhost/"
    assert provider_config.openai_config.api_key == "secret_key"

    # configuration for other providers must not be set
    assert provider_config.rhoai_vllm_config is None
    assert provider_config.rhelai_vllm_config is None
    assert provider_config.azure_config is None
    assert provider_config.watsonx_config is None
    assert provider_config.bam_config is None


def test_provider_config_openai_unknown_parameters():
    """Test if unknown OpenAI parameters are detected."""
    # provider type is set to "openai" and OpenAI-specific configuration is there
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "openai",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "deployment_name": "deploment-name",
                "openai_config": {
                    "unknown_parameter": "unknown value",
                    "url": "http://localhost",
                    "tenant_id": "tenant-ID",
                    "client_id": "client-ID",
                    "client_secret_path": "tests/config/secret/apitoken",
                    "credentials_path": "tests/config/secret/apitoken",
                    "deployment_name": "deployment-name",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


def test_provider_config_rhoai_vllm_specific():
    """Test if RHOAI VLLM-specific config is loaded and validated."""
    # provider type is set to "rhoai_vllm" and RHOAI VLLM-specific configuration is there
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": "rhoai_vllm",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "test_project_id",
            "rhoai_vllm_config": {
                "url": "http://localhost",
                "credentials_path": "tests/config/secret/apitoken",
            },
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )
    # OpenAI-specific configuration must be present
    assert provider_config.rhoai_vllm_config is not None
    assert str(provider_config.rhoai_vllm_config.url) == "http://localhost/"
    assert provider_config.rhoai_vllm_config.api_key == "secret_key"
    assert provider_config.certificates_store == "/tmp/ols.pem"  # noqa: S108

    # configuration for other providers must not be set
    assert provider_config.openai_config is None
    assert provider_config.azure_config is None
    assert provider_config.watsonx_config is None
    assert provider_config.bam_config is None


def test_provider_config_rhoai_vllm_unknown_parameters():
    """Test if unknown RHOAI-VLLM parameters are detected."""
    # provider type is set to "rhoai_vllm" and RHOAI VLLM-specific configuration is there
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "rhoai_vllm",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "deployment_name": "deploment-name",
                "rhoai_vllm_config": {
                    "unknown_parameter": "unknown value",
                    "url": "http://localhost",
                    "tenant_id": "tenant-ID",
                    "client_id": "client-ID",
                    "client_secret_path": "tests/config/secret/apitoken",
                    "credentials_path": "tests/config/secret/apitoken",
                    "deployment_name": "deployment-name",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


def test_provider_config_rhelai_vllm_specific():
    """Test if RHELAI VLLM-specific config is loaded and validated."""
    # provider type is set to "rhelai_vllm" and RHELAI VLLM-specific configuration is there
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": "rhelai_vllm",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "test_project_id",
            "rhelai_vllm_config": {
                "url": "http://localhost",
                "credentials_path": "tests/config/secret/apitoken",
            },
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )
    # RHELAI VLLM-specific configuration must be present
    assert provider_config.rhelai_vllm_config is not None
    assert str(provider_config.rhelai_vllm_config.url) == "http://localhost/"
    assert provider_config.rhelai_vllm_config.api_key == "secret_key"
    assert provider_config.certificates_store == "/tmp/ols.pem"  # noqa: S108

    # configuration for other providers must not be set
    assert provider_config.rhoai_vllm_config is None
    assert provider_config.openai_config is None
    assert provider_config.azure_config is None
    assert provider_config.watsonx_config is None
    assert provider_config.bam_config is None


def test_provider_config_rhelai_vllm_unknown_parameters():
    """Test if unknown RHELAI-VLLM parameters are detected."""
    # provider type is set to "rhelai_vllm" and RHELAI VLLM-specific configuration is there
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "rhelai_vllm",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "deployment_name": "deploment-name",
                "rhelai_vllm_config": {
                    "unknown_parameter": "unknown value",
                    "url": "http://localhost",
                    "tenant_id": "tenant-ID",
                    "client_id": "client-ID",
                    "client_secret_path": "tests/config/secret/apitoken",
                    "credentials_path": "tests/config/secret/apitoken",
                    "deployment_name": "deployment-name",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


def test_provider_config_watsonx_specific():
    """Test if Watsonx-specific config is loaded and validated."""
    # provider type is set to "watsonx" and Watsonx-specific configuration is there
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": "watsonx",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "test_project_id",
            "watsonx_config": {
                "url": "http://localhost",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "*project id*",
            },
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )
    # Watsonx-specific configuration must be present
    assert provider_config.watsonx_config is not None
    assert str(provider_config.watsonx_config.url) == "http://localhost/"
    assert provider_config.watsonx_config.project_id == "*project id*"
    assert provider_config.watsonx_config.api_key == "secret_key"

    # configuration for other providers must not be set
    assert provider_config.rhoai_vllm_config is None
    assert provider_config.rhelai_vllm_config is None
    assert provider_config.azure_config is None
    assert provider_config.openai_config is None
    assert provider_config.bam_config is None

    assert provider_config.api_version is None


def test_provider_config_watsonx_unknown_parameters():
    """Test if unknown Watsonx parameters are detected."""
    # provider type is set to "watsonx" and Watsonx-specific configuration is there
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "watsonx",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "watsonx_config": {
                    "unknown_parameter": "unknown value",
                    "url": "http://localhost",
                    "credentials_path": "tests/config/secret/apitoken",
                    "project_id": "*project id*",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


def test_provider_config_bam_specific():
    """Test if BAM-specific config is loaded and validated."""
    # provider type is set to "bam" and BAM-specific configuration is there
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": "bam",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "test_project_id",
            "bam_config": {
                "url": "http://localhost",
                "credentials_path": "tests/config/secret/apitoken",
            },
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )
    # BAM-specific configuration must be present
    assert provider_config.bam_config is not None
    assert str(provider_config.bam_config.url) == "http://localhost/"
    assert provider_config.bam_config.api_key == "secret_key"

    # configuration for other providers must not be set
    assert provider_config.rhoai_vllm_config is None
    assert provider_config.rhelai_vllm_config is None
    assert provider_config.azure_config is None
    assert provider_config.openai_config is None
    assert provider_config.watsonx_config is None


def test_provider_config_bam_unknown_parameters():
    """Test if unknown BAM parameters are detected."""
    # provider type is set to "bam" and BAM-specific configuration is there
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "bam_config": {
                    "unknown_parameter": "unknown value",
                    "url": "http://localhost",
                    "credentials_path": "tests/config/secret/apitoken",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


def test_improper_provider_specific_config():
    """Test if check for improper provider-specific config is performed."""
    with pytest.raises(
        InvalidConfigurationError,
        match="provider type bam selected, but configuration is set for different provider",
    ):
        # provider type is set to "bam" but OpenAI-specific configuration is there
        ProviderConfig(
            {
                "name": "test_name",
                "type": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "openai_config": {
                    "url": "http://localhost",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


def test_multiple_provider_specific_configs():
    """Test if check for multiple provider-specific configs is performed."""
    with pytest.raises(
        InvalidConfigurationError,
        match="multiple provider-specific configurations found, but just one is expected for provider bam",  # noqa: E501
    ):
        # two provider-specific configurations is in the configuration
        ProviderConfig(
            {
                "name": "test_name",
                "type": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "openai_config": {
                    "url": "http://localhost",
                },
                "watsonx_config": {
                    "url": "http://localhost",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


providers = (
    constants.PROVIDER_BAM,
    constants.PROVIDER_OPENAI,
    constants.PROVIDER_AZURE_OPENAI,
    constants.PROVIDER_WATSONX,
    constants.PROVIDER_RHOAI_VLLM,
    constants.PROVIDER_RHELAI_VLLM,
)


@pytest.mark.parametrize("provider_name", providers)
def test_provider_model_default_tokens_limit(provider_name):
    """Test if the token limits are set as default when not set."""
    # provider config with attributes 'blended' for all providers
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": provider_name,
            "url": "test_url",
            "deployment_name": "test",
            "project_id": 42,
            "models": [
                {
                    "name": "test_model_name",
                }
            ],
        }
    )
    # expected token limit for given model, default is used if not set.
    expected_limit = constants.DEFAULT_CONTEXT_WINDOW_SIZE

    assert (
        provider_config.models["test_model_name"].context_window_size == expected_limit
    )


def test_provider_config_explicit_tokens():
    """Test the ProviderConfig model when explicit tokens are specified."""
    # Note: context window should be >= 4096 (default) response token limit
    context_window_size = 4097

    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": "bam",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "test_project_id",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.url/",
                    "credentials_path": "tests/config/secret/apitoken",
                    "context_window_size": context_window_size,
                }
            ],
        }
    )
    assert (
        provider_config.models["test_model_name"].context_window_size
        == context_window_size
    )


def test_provider_config_improper_context_window_size_value():
    """Test the ProviderConfig model when improper context window size is specified."""
    with pytest.raises(
        ValidationError,
        match="Input should be greater than 0",
    ):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                        "context_window_size": -1,
                    }
                ],
            }
        )


def test_provider_config_improper_context_window_size_type():
    """Test the ProviderConfig model when improper context window size is specified."""
    with pytest.raises(
        ValidationError,
        match="Input should be a valid integer, unable to parse string as an integer",
    ):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                        "context_window_size": "not-a-number",
                    }
                ],
            }
        )


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
            "name": "bam",
            "url": "http://test.url",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.model.url",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )

    provider_config.validate_yaml()


def test_provider_config_validation_improper_url():
    """Test the ProviderConfig model validation for improper URL."""
    provider_config = ProviderConfig(
        {
            "name": "bam",
            "url": "httpXXX://test.url",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.model.url",
                    "credentials_path": "tests/config/secret/apitoken",
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
            "type": "bam",
            "url": "httpXXX://test.url",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.model.url",
                    "credentials_path": "tests/config/secret/apitoken",
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
            "name": "bam",
            "url": "http://test.url",
            "credentials_path": None,
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.model.url",
                    "credentials_path": "tests/config/secret/apitoken",
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
                "type": "bam",
                "url": "test_provider_url",
                "credentials_path": "tests/config/secret/apitoken",
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "http://test.url/",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            },
        ]
    )
    assert len(llm_providers.providers) == 1
    assert llm_providers.providers["test_provider_name"].name == "test_provider_name"
    assert llm_providers.providers["test_provider_name"].type == "bam"
    assert llm_providers.providers["test_provider_name"].url == "test_provider_url"
    assert llm_providers.providers["test_provider_name"].credentials == "secret_key"
    assert len(llm_providers.providers["test_provider_name"].models) == 1
    assert (
        llm_providers.providers["test_provider_name"].models["test_model_name"].name
        == "test_model_name"
    )
    assert (
        str(llm_providers.providers["test_provider_name"].models["test_model_name"].url)
        == "http://test.url/"
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
                    "credentials_path": "tests/config/secret/apitoken",
                    "models": [],
                },
            ]
        )
    assert "provider name is missing" in str(excinfo.value)


def test_llm_providers_type_defaulting():
    """Test that provider type is defaulted from provider name."""
    llm_providers = LLMProviders(
        [
            {
                "name": "bam",
                "models": [
                    {
                        "name": "m1",
                        "url": "https://test_model_url",
                    }
                ],
            },
        ]
    )
    assert len(llm_providers.providers) == 1
    assert llm_providers.providers["bam"].name == "bam"
    assert llm_providers.providers["bam"].type == "bam"

    llm_providers = LLMProviders(
        [
            {
                "name": "test_provider",
                "type": "bam",
                "models": [
                    {
                        "name": "m1",
                        "url": "https://test_model_url",
                    }
                ],
            },
        ]
    )
    assert len(llm_providers.providers) == 1
    assert llm_providers.providers["test_provider"].name == "test_provider"
    assert llm_providers.providers["test_provider"].type == "bam"


def test_llm_providers_type_validation():
    """Test that only known provider types are allowed."""
    with pytest.raises(InvalidConfigurationError) as excinfo:
        LLMProviders(
            [
                {
                    "name": "invalid_provider",
                },
            ]
        )
    assert "invalid provider type: invalid_provider" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        LLMProviders(
            [
                {"name": "bam", "type": "invalid_type"},
            ]
        )
    assert "invalid provider type: invalid_type" in str(excinfo.value)


def test_llm_providers_watsonx_required_projectid():
    """Test that project_id is required for Watsonx provider."""
    with pytest.raises(InvalidConfigurationError) as excinfo:
        LLMProviders(
            [
                {
                    "name": "watsonx",
                },
            ]
        )
    assert "project_id is required for Watsonx provider" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        LLMProviders(
            [
                {
                    "name": "test_watsonx",
                    "type": "watsonx",
                },
            ]
        )
    assert "project_id is required for Watsonx provider" in str(excinfo.value)

    llm_providers = LLMProviders(
        [
            {
                "name": "watsonx",
                "project_id": "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX",
                "models": [
                    {
                        "name": "m1",
                        "url": "https://test_model_url",
                    }
                ],
            },
        ]
    )
    assert len(llm_providers.providers) == 1
    assert llm_providers.providers["watsonx"].name == "watsonx"
    assert llm_providers.providers["watsonx"].type == "watsonx"
    assert (
        llm_providers.providers["watsonx"].project_id
        == "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
    )

    llm_providers = LLMProviders(
        [
            {
                "name": "test_provider",
                "type": "watsonx",
                "project_id": "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX",
                "models": [
                    {"name": "test_model_name", "url": "http://test_model_url/"}
                ],
            },
        ]
    )
    assert len(llm_providers.providers) == 1
    assert llm_providers.providers["test_provider"].name == "test_provider"
    assert llm_providers.providers["test_provider"].type == "watsonx"
    assert (
        llm_providers.providers["test_provider"].project_id
        == "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
    )


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


def test_valid_values():
    """Test valid values."""
    # test default values
    logging_config = LoggingConfig()
    assert logging_config.app_log_level == logging.INFO
    assert logging_config.lib_log_level == logging.WARNING
    assert logging_config.uvicorn_log_level == logging.WARNING
    assert not logging_config.suppress_metrics_in_log
    assert not logging_config.suppress_auth_checks_warning_in_log

    # test custom values
    logging_config = LoggingConfig(
        app_log_level="debug",
        lib_log_level="debug",
        uvicorn_log_level="debug",
    )
    assert logging_config.app_log_level == logging.DEBUG
    assert logging_config.lib_log_level == logging.DEBUG
    assert logging_config.uvicorn_log_level == logging.DEBUG

    logging_config = LoggingConfig()
    assert logging_config.app_log_level == logging.INFO


def test_invalid_values():
    """Test invalid values."""
    # value is not string
    with pytest.raises(
        InvalidConfigurationError,
        match="'5' log level must be string, got <class 'int'>",
    ):
        LoggingConfig(app_log_level=5)

    # value is not valid log level
    with pytest.raises(
        InvalidConfigurationError,
        match="'dingdong' is not valid log level, valid levels are",
    ):
        LoggingConfig(app_log_level="dingdong")

    # value is not valid log level
    with pytest.raises(
        InvalidConfigurationError,
        match="'foo' is not valid log level, valid levels are",
    ):
        LoggingConfig(uvicorn_log_level="foo")


def test_tls_security_profile_default_values():
    """Test the TLSSecurityProfile model."""
    tls_security_profile = TLSSecurityProfile()
    assert tls_security_profile.profile_type is None
    assert tls_security_profile.min_tls_version is None
    assert tls_security_profile.ciphers is None


def test_tls_security_profile_correct_values():
    """Test the TLSSecurityProfile model."""
    tls_security_profile = TLSSecurityProfile(
        {
            "type": "Custom",
            "minTLSVersion": "VersionTLS13",
            "ciphers": [
                "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
                "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
            ],
        }
    )
    assert tls_security_profile.profile_type == "Custom"
    assert tls_security_profile.min_tls_version == "VersionTLS13"
    assert "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256" in tls_security_profile.ciphers
    assert "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384" in tls_security_profile.ciphers


tls_types = (
    tls.TLSProfiles.OLD_TYPE,
    tls.TLSProfiles.INTERMEDIATE_TYPE,
    tls.TLSProfiles.MODERN_TYPE,
    tls.TLSProfiles.CUSTOM_TYPE,
)


tls_versions = (
    tls.TLSProtocolVersion.VERSION_TLS_10,
    tls.TLSProtocolVersion.VERSION_TLS_11,
    tls.TLSProtocolVersion.VERSION_TLS_12,
    tls.TLSProtocolVersion.VERSION_TLS_13,
)


@pytest.mark.parametrize("tls_type", tls_types)
@pytest.mark.parametrize("min_tls_version", tls_versions)
def test_tls_security_profile_validate_yaml(tls_type, min_tls_version):
    """Test the TLSSecurityProfile model validation."""
    tls_security_profile = TLSSecurityProfile()
    tls_security_profile.validate_yaml()

    tls_security_profile = TLSSecurityProfile(
        {
            "type": tls_type,
            "minTLSVersion": min_tls_version,
            "ciphers": [],
        }
    )
    tls_security_profile.validate_yaml()


def test_tls_security_profile_validate_invalid_yaml_type():
    """Test the TLSSecurityProfile model validation."""
    tls_security_profile = TLSSecurityProfile(
        {
            "type": "Foo",
            "minTLSVersion": "VersionTLS13",
            "ciphers": [
                "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
                "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
            ],
        }
    )
    with pytest.raises(
        InvalidConfigurationError,
        match="Invalid TLS profile type 'Foo'",
    ):
        tls_security_profile.validate_yaml()


def test_tls_security_profile_validate_invalid_yaml_min_tls_version():
    """Test the TLSSecurityProfile model validation."""
    tls_security_profile = TLSSecurityProfile(
        {
            "type": "Custom",
            "minTLSVersion": "foo",
            "ciphers": [
                "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
                "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
            ],
        }
    )
    with pytest.raises(
        InvalidConfigurationError,
        match="Invalid minimal TLS version 'foo'",
    ):
        tls_security_profile.validate_yaml()


tls_types_without_custom = (
    tls.TLSProfiles.OLD_TYPE,
    tls.TLSProfiles.INTERMEDIATE_TYPE,
    tls.TLSProfiles.MODERN_TYPE,
)


@pytest.mark.parametrize("tls_type", tls_types_without_custom)
def test_tls_security_profile_validate_invalid_yaml_ciphers(tls_type):
    """Test the TLSSecurityProfile model validation."""
    tls_security_profile = TLSSecurityProfile(
        {
            "type": tls_type,
            "minTLSVersion": "VersionTLS13",
            "ciphers": [
                "foo",
                "bar",
            ],
        }
    )
    with pytest.raises(
        InvalidConfigurationError,
        match="Unsupported cipher 'foo' found in configuration",
    ):
        tls_security_profile.validate_yaml()


def test_tls_security_profile_equality():
    """Test equality or inequality of two security profiles."""
    profile1 = TLSSecurityProfile(
        {
            "type": "Custom",
            "minTLSVersion": "VersionTLS13",
            "ciphers": [],
        }
    )
    profile2 = TLSSecurityProfile(
        {
            "type": "Custom",
            "minTLSVersion": "VersionTLS13",
            "ciphers": [],
        }
    )
    assert profile1 == profile2

    profile3 = TLSSecurityProfile(
        {
            "type": "Custom",
            "minTLSVersion": "VersionTLS12",
            "ciphers": [],
        }
    )
    assert profile1 != profile3


def test_tls_config_default_values():
    """Test the TLSConfig model."""
    tls_config = TLSConfig()
    assert tls_config.tls_certificate_path is None
    assert tls_config.tls_key_path is None
    assert tls_config.tls_key_password is None


def test_tls_config_correct_values():
    """Test the TLSConfig model."""
    tls_config = TLSConfig(
        {
            "tls_certificate_path": "tests/config/empty_cert.crt",
            "tls_key_path": "tests/config/key",
            "tls_key_password_path": "tests/config/password",
        }
    )
    assert tls_config.tls_certificate_path == "tests/config/empty_cert.crt"
    assert tls_config.tls_key_path == "tests/config/key"
    assert tls_config.tls_key_password == "* this is password *"  # noqa: S105
    tls_config.validate_yaml(False)


def test_tls_config_incorrect_password_path():
    """Test the TLSConfig model."""
    with pytest.raises(FileNotFoundError, match="No such file"):
        TLSConfig(
            {
                "tls_certificate_path": "tests/config/empty_cert.crt",
                "tls_key_path": "tests/config/key",
                "tls_key_password_path": "this/file/does/not/exist",
            }
        )
    with pytest.raises(IsADirectoryError, match="Is a directory"):
        TLSConfig(
            {
                "tls_certificate_path": "tests/config/empty_cert.crt",
                "tls_key_path": "tests/config/key",
                "tls_key_password_path": "/",
            }
        )


def test_tls_config_incorrect_certificate_path():
    """Test the TLSConfig model with incorrect path to certificate."""
    config = TLSConfig(
        {
            "tls_certificate_path": "/",
            "tls_key_path": "tests/config/key",
            "tls_key_password_path": "tests/config/password",
        }
    )
    with pytest.raises(InvalidConfigurationError, match="is not a file"):
        config.validate_yaml()

    config2 = TLSConfig(
        {
            "tls_certificate_path": "/etc/shadow",
            "tls_key_path": "tests/config/key",
            "tls_key_password_path": "tests/config/password",
        }
    )
    with pytest.raises(InvalidConfigurationError, match="is not readable"):
        config2.validate_yaml()


def test_tls_config_no_data_provided():
    """Test the TLSConfig model."""
    tls_config = TLSConfig(None)
    with pytest.raises(
        InvalidConfigurationError,
        match="Can not enable TLS without ols_config.tls_config.tls_certificate_path",
    ):
        tls_config.validate_yaml(False)


def test_tls_config_no_tls_key_path():
    """Test the TLSConfig model if tls_key_path is not specified."""
    tls_config = TLSConfig(
        {
            "tls_certificate_path": "tests/config/empty_cert.crt",
            "tls_key_password_path": "tests/config/password",
        }
    )
    with pytest.raises(
        InvalidConfigurationError,
        match="Can not enable TLS without ols_config.tls_config.tls_key_path",
    ):
        tls_config.validate_yaml(False)


def test_postgres_config_default_values():
    """Test the PostgresConfig model."""
    postgres_config = PostgresConfig()
    assert postgres_config.host == constants.POSTGRES_CACHE_HOST
    assert postgres_config.port == constants.POSTGRES_CACHE_PORT
    assert postgres_config.dbname == constants.POSTGRES_CACHE_DBNAME
    assert postgres_config.user == constants.POSTGRES_CACHE_USER
    assert postgres_config.max_entries == constants.POSTGRES_CACHE_MAX_ENTRIES


def test_postgres_config_correct_values():
    """Test the PostgresConfig model when correct values are used."""
    postgres_config = PostgresConfig(
        host="other_host",
        port=1234,
        dbname="my_database",
        user="admin",
        ssl_mode="allow",
        max_entries=42,
    )

    # explicitly set values
    assert postgres_config.host == "other_host"
    assert postgres_config.port == 1234
    assert postgres_config.dbname == "my_database"
    assert postgres_config.user == "admin"
    assert postgres_config.ssl_mode == "allow"
    assert postgres_config.max_entries == 42


def test_postgres_config_wrong_port():
    """Test the PostgresConfig model."""
    with pytest.raises(
        ValidationError, match="The port needs to be between 0 and 65536"
    ):
        PostgresConfig(
            host="other_host",
            port=9999999,
            dbname="my_database",
            user="admin",
            ssl_mode="allow",
        )


def test_postgres_config_equality():
    """Test the PostgresConfig equality check."""
    postgres_config_1 = PostgresConfig()
    postgres_config_2 = PostgresConfig()

    # compare the same Postgres configs
    assert postgres_config_1 == postgres_config_2

    # compare different Postgres configs
    postgres_config_2.host = "12.34.56.78"
    assert postgres_config_1 != postgres_config_2

    # compare with value of different type
    other_value = "foo"
    assert postgres_config_1 != other_value


def test_postgres_config_with_password():
    """Test the PostgresConfig model."""
    postgres_config = PostgresConfig(
        host="other_host",
        port=1234,
        dbname="my_database",
        user="admin",
        password_path="tests/config/postgres_password.txt",  # noqa: S106
        ssl_mode="allow",
        max_entries=42,
    )
    # check if password was read correctly from file
    assert postgres_config.password == "postgres_password"  # noqa: S105


def test_redis_config():
    """Test the RedisConfig model."""
    redis_config = RedisConfig({})
    # default values
    assert redis_config.retry_on_error == constants.REDIS_RETRY_ON_ERROR
    assert redis_config.retry_on_timeout == constants.REDIS_RETRY_ON_TIMEOUT
    assert redis_config.number_of_retries == constants.REDIS_NUMBER_OF_RETRIES

    redis_config = RedisConfig(
        {
            "host": "localhost",
            "port": 6379,
            "max_memory": "200mb",
            "max_memory_policy": "allkeys-lru",
            "retry_on_error": "false",
            "retry_on_timeout": "false",
            "number_of_retries": 42,
        }
    )

    # explicitly set values
    assert redis_config.host == "localhost"
    assert redis_config.port == 6379
    assert redis_config.max_memory == "200mb"
    assert redis_config.max_memory_policy == "allkeys-lru"
    assert redis_config.retry_on_error is False
    assert redis_config.retry_on_timeout is False
    assert redis_config.number_of_retries == 42

    redis_config = RedisConfig(
        {
            "host": "localhost",
            "port": 6379,
            "max_memory": "200mb",
            "max_memory_policy": "allkeys-lru",
            "retry_on_error": "true",
            "retry_on_timeout": "true",
            "number_of_retries": 100,
        }
    )
    assert redis_config.host == "localhost"
    assert redis_config.port == 6379
    assert redis_config.max_memory == "200mb"
    assert redis_config.max_memory_policy == "allkeys-lru"
    assert redis_config.retry_on_error is True
    assert redis_config.retry_on_timeout is True
    assert redis_config.number_of_retries == 100

    redis_config = RedisConfig()

    # initial values
    assert redis_config.host is None
    assert redis_config.port is None
    assert redis_config.max_memory is None
    assert redis_config.max_memory_policy is None
    assert redis_config.retry_on_error is None
    assert redis_config.retry_on_timeout is None
    assert redis_config.number_of_retries is None


def test_redis_config_with_ca_cert_path():
    """Test the RedisConfig model with CA certificate path."""
    redis_config = RedisConfig(
        {
            "host": "localhost",
            "port": 6379,
            "max_memory": "200mb",
            "max_memory_policy": "allkeys-lru",
            "ca_cert_path": "tests/config/redis_ca_cert.crt",
        }
    )
    assert redis_config.ca_cert_path == "tests/config/redis_ca_cert.crt"


def test_redis_config_with_no_ca_cert_path():
    """Test the RedisConfig model with no CA certificate path."""
    redis_config = RedisConfig(
        {
            "host": "localhost",
            "port": 6379,
            "max_memory": "200mb",
            "max_memory_policy": "allkeys-lru",
        }
    )
    assert redis_config.ca_cert_path is None


def test_redis_config_with_password():
    """Test the RedisConfig model."""
    redis_config = RedisConfig(
        {
            "host": "localhost",
            "port": 6379,
            "max_memory": "200mb",
            "max_memory_policy": "allkeys-lru",
            "password_path": "tests/config/redis_password.txt",
        }
    )
    assert redis_config.password == "redis_password"  # noqa: S105


def test_redis_config_with_no_password():
    """Test the RedisConfig model with no password."""
    redis_config = RedisConfig(
        {
            "host": "localhost",
            "port": 6379,
            "max_memory": "200mb",
            "max_memory_policy": "allkeys-lru",
        }
    )
    assert redis_config.password is None


def test_redis_config_with_invalid_password_path():
    """Test the RedisConfig model with invalid password path."""
    with pytest.raises(NotADirectoryError, match="Not a directory"):
        RedisConfig(
            {
                "host": "localhost",
                "port": 6379,
                "max_memory": "200mb",
                "max_memory_policy": "allkeys-lru",
                "password_path": "/dev/null/foobar",
            }
        )


def test_redis_config_invalid_port():
    """Test the RedisConfig model with invalid password path."""
    with pytest.raises(InvalidConfigurationError):
        RedisConfig(
            {
                "host": "localhost",
                "port": -1,
                "max_memory": "200mb",
                "max_memory_policy": "allkeys-lru",
            }
        )

    with pytest.raises(InvalidConfigurationError):
        RedisConfig(
            {
                "host": "localhost",
                "port": 100000,
                "max_memory": "200mb",
                "max_memory_policy": "allkeys-lru",
            }
        )


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


def test_redis_config_yaml_valiation():
    """Test the RedisConfig yaml validation method."""
    redis_config = RedisConfig(
        {
            "host": "localhost",
            "port": 6379,
            "max_memory": "200mb",
            "max_memory_policy": "allkeys-lru",
            "retry_on_error": "false",
            "retry_on_timeout": "false",
            "number_of_retries": 42,
        }
    )
    redis_config.validate_yaml()

    # change max_memory_policy
    redis_config.max_memory_policy = "allkeys-lru"
    redis_config.validate_yaml()

    # change max_memory_policy
    redis_config.max_memory_policy = "volatile-lru"
    redis_config.validate_yaml()

    # unknown max_memory_policy
    # -> it should raises an exception
    redis_config.max_memory_policy = "unknown"
    with pytest.raises(InvalidConfigurationError):
        redis_config.validate_yaml()


def test_memory_cache_config():
    """Test the MemoryCacheConfig model."""
    memory_cache_config = InMemoryCacheConfig(
        {
            "max_entries": 100,
        }
    )
    assert memory_cache_config.max_entries == 100

    memory_cache_config = InMemoryCacheConfig()
    assert memory_cache_config.max_entries is None


def test_memory_cache_config_improper_entries():
    """Test the MemoryCacheConfig model if improper max_entries is used."""
    with pytest.raises(
        InvalidConfigurationError,
        match="invalid max_entries for memory conversation cache",
    ):
        InMemoryCacheConfig(
            {
                "max_entries": -100,
            }
        )


def test_memory_config_equality():
    """Test the MemoryConfig equality check."""
    memory_config_1 = InMemoryCacheConfig()
    memory_config_2 = InMemoryCacheConfig()

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

    conversation_cache_config = ConversationCacheConfig(
        {
            "type": "postgres",
            "postgres": {
                "host": "1.2.3.4",
                "port": 1234,
                "dbname": "testdb",
                "user": "user",
                "ssl_mode": "allow",
            },
        }
    )
    assert conversation_cache_config.type == "postgres"
    assert conversation_cache_config.postgres.host == "1.2.3.4"
    assert conversation_cache_config.postgres.port == 1234
    assert conversation_cache_config.postgres.dbname == "testdb"
    assert conversation_cache_config.postgres.user == "user"
    assert conversation_cache_config.postgres.ssl_mode == "allow"

    conversation_cache_config = ConversationCacheConfig()
    assert conversation_cache_config.type is None
    assert conversation_cache_config.redis is None
    assert conversation_cache_config.memory is None
    assert conversation_cache_config.postgres is None

    with pytest.raises(
        InvalidConfigurationError,
        match="redis conversation cache type is specified, but redis configuration is missing",
    ):
        ConversationCacheConfig({"type": "redis"})

    with pytest.raises(
        InvalidConfigurationError,
        match="memory conversation cache type is specified, but memory configuration is missing",
    ):
        ConversationCacheConfig({"type": "memory"})

    with pytest.raises(
        InvalidConfigurationError,
        match="Postgres conversation cache type is specified, but Postgres configuration is missing",  # noqa: E501
    ):
        ConversationCacheConfig({"type": "postgres"})


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


def test_conversation_cache_config_equality():
    """Test the ConversationCacheConfig equality check."""
    conversation_cache_config_1 = ConversationCacheConfig()
    conversation_cache_config_2 = ConversationCacheConfig()

    # compare the same conversation_cache configs
    assert conversation_cache_config_1 == conversation_cache_config_2

    # compare different conversation_cache configs
    conversation_cache_config_2.type = "some non-default type"
    assert conversation_cache_config_1 != conversation_cache_config_2

    # compare with value of different type
    other_value = "foo"
    assert conversation_cache_config_1 != other_value


def test_ols_config(tmpdir):
    """Test the OLSConfig model."""
    ols_config = OLSConfig(
        {
            "default_provider": "test_default_provider",
            "default_model": "test_default_model",
            "introspection_enabled": True,
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
    assert ols_config.introspection_enabled
    assert (
        ols_config.query_validation_method == constants.QueryValidationMethod.DISABLED
    )
    assert ols_config.user_data_collection == UserDataCollection()
    assert ols_config.reference_content is None
    assert ols_config.authentication_config == AuthenticationConfig(
        module=constants.DEFAULT_AUTHENTICATION_MODULE
    )
    assert ols_config.extra_ca == []
    assert ols_config.certificate_directory == constants.DEFAULT_CERTIFICATE_DIRECTORY
    assert ols_config.system_prompt_path is None
    assert ols_config.system_prompt is None
    assert ols_config.tls_security_profile == TLSSecurityProfile()


def test_ols_config_with_auth_config(tmpdir):
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
            "authentication_config": {"module": "foo"},
        }
    )
    assert ols_config.default_provider == "test_default_provider"
    assert ols_config.default_model == "test_default_model"
    assert ols_config.conversation_cache.type == "memory"
    assert ols_config.conversation_cache.memory.max_entries == 100
    assert not ols_config.introspection_enabled
    assert ols_config.logging_config.app_log_level == logging.INFO
    assert (
        ols_config.query_validation_method == constants.QueryValidationMethod.DISABLED
    )
    assert ols_config.user_data_collection == UserDataCollection()
    assert ols_config.reference_content is None
    assert ols_config.authentication_config == AuthenticationConfig(module="foo")
    assert ols_config.extra_ca == []
    assert ols_config.certificate_directory == constants.DEFAULT_CERTIFICATE_DIRECTORY
    assert ols_config.system_prompt_path is None
    assert ols_config.system_prompt is None
    assert ols_config.tls_security_profile == TLSSecurityProfile()


def test_ols_config_with_tls_security_profile(tmpdir):
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
            "tlsSecurityProfile": {
                "type": "Custom",
                "minTLSVersion": "VersionTLS13",
                "ciphers": [
                    "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
                    "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
                ],
            },
        }
    )
    assert ols_config.default_provider == "test_default_provider"
    assert ols_config.default_model == "test_default_model"
    assert ols_config.conversation_cache.type == "memory"
    assert ols_config.conversation_cache.memory.max_entries == 100
    assert ols_config.logging_config.app_log_level == logging.INFO
    assert (
        ols_config.query_validation_method == constants.QueryValidationMethod.DISABLED
    )
    assert ols_config.user_data_collection == UserDataCollection()
    assert ols_config.reference_content is None
    assert ols_config.authentication_config == AuthenticationConfig(
        module=constants.DEFAULT_AUTHENTICATION_MODULE
    )
    assert ols_config.extra_ca == []
    assert ols_config.certificate_directory == constants.DEFAULT_CERTIFICATE_DIRECTORY
    assert ols_config.system_prompt_path is None
    assert ols_config.system_prompt is None
    assert ols_config.tls_security_profile is not None
    assert ols_config.tls_security_profile.profile_type == "Custom"
    assert ols_config.tls_security_profile.min_tls_version == "VersionTLS13"
    assert (
        "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
        in ols_config.tls_security_profile.ciphers
    )
    assert (
        "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"
        in ols_config.tls_security_profile.ciphers
    )


def get_ols_configs():
    """Construct two instances of OLSConfig class."""
    ols_config_1 = OLSConfig()
    ols_config_2 = OLSConfig()
    return ols_config_1, ols_config_2


def test_ols_config_equality(subtests):
    """Check the equality operator of OLSConfig class."""
    # default configurations
    ols_config_1, ols_config_2 = get_ols_configs()

    # compare the same configs
    assert ols_config_1 == ols_config_2

    # compare two objects with different content (attributes)

    # system_prompt attribute (str)
    with subtests.test(msg="Different attribute: system_prompt"):
        ols_config_1, ols_config_2 = get_ols_configs()
        ols_config_1.system_prompt = "new system prompt"
        assert ols_config_1 != ols_config_2

    # system_prompt_path attribute (str)
    with subtests.test(msg="Different attribute: system_prompt_path"):
        ols_config_1, ols_config_2 = get_ols_configs()
        ols_config_1.system_prompt_path = "foo"
        assert ols_config_1 != ols_config_2

    # default_provider attribute (str)
    with subtests.test(msg="Different attribute: default_provider"):
        ols_config_1, ols_config_2 = get_ols_configs()
        ols_config_1.default_provider = "my_own_provider"
        assert ols_config_1 != ols_config_2

    # default_model attribute (str)
    with subtests.test(msg="Different attribute: default_model"):
        ols_config_1, ols_config_2 = get_ols_configs()
        ols_config_1.default_model = "my_own_model"
        assert ols_config_1 != ols_config_2

    # certificate_directory attribute (str/dir)
    with subtests.test(msg="Different attribute: certificate_directory"):
        ols_config_1, ols_config_2 = get_ols_configs()
        ols_config_1.certificate_directory = "/dev/none"
        assert ols_config_1 != ols_config_2

    # max_workers attribute (int)
    with subtests.test(msg="Different attribute: max_workers"):
        ols_config_1, ols_config_2 = get_ols_configs()
        ols_config_1.max_workers = 42
        assert ols_config_1 != ols_config_2

    # expire_llm_is_ready_persistent_state attribute (int)
    with subtests.test(msg="Different attribute: expire_llm_is_ready_persistent_state"):
        ols_config_1, ols_config_2 = get_ols_configs()
        ols_config_1.expire_llm_is_ready_persistent_state = 42
        assert ols_config_1 != ols_config_2

    # logging_config attribute (LoggingConfig)
    with subtests.test(msg="Different attribute: logging_config"):
        ols_config_1, ols_config_2 = get_ols_configs()
        ols_config_1.logging_config = LoggingConfig()
        assert ols_config_1 != ols_config_2

    # reference_content attribute (ReferenceContent)
    with subtests.test(msg="Different attribute: reference_content"):
        ols_config_1, ols_config_2 = get_ols_configs()
        ols_config_1.reference_content = ReferenceContent()
        assert ols_config_1 != ols_config_2

    # authentication_config attribute (AuthenticationConfig)
    with subtests.test(msg="Different attribute: authentication_config"):
        ols_config_1, ols_config_2 = get_ols_configs()
        ols_config_1.authentication_config = AuthenticationConfig(
            skip_tls_verification=True
        )
        assert ols_config_1 != ols_config_2

    # tls_config attribute (TLSConfig)
    with subtests.test(msg="Different attribute: tls_config"):
        ols_config_1, ols_config_2 = get_ols_configs()
        ols_config_1.tls_config = TLSConfig(
            {
                "tls_certificate_path": "tests/config/empty_cert.crt",
                "tls_key_path": "tests/config/key",
                "tls_key_password_path": "tests/config/password",
            }
        )
        assert ols_config_1 != ols_config_2

    # conversation_cache attribute (ConversationCacheConfig)
    with subtests.test(msg="Different attribute: conversation_cache_config"):
        ols_config_1, ols_config_2 = get_ols_configs()
        ols_config_1.conversation_cache = ConversationCacheConfig()
        assert ols_config_1 != ols_config_2

    # tls_security_profile attribute (TLSSecurityProfile)
    with subtests.test(msg="Different attribute: conversation_cache_config"):
        ols_config_1, ols_config_2 = get_ols_configs()
        ols_config_1.tls_security_profile = TLSSecurityProfile()
        assert ols_config_1 != ols_config_2

    # quota handler attribute (QuotaHandlersConfig)
    with subtests.test(msg="Different attribute: quota_handlers"):
        ols_config_1, ols_config_2 = get_ols_configs()
        ols_config_1.quota_handlers = QuotaHandlersConfig()
        assert ols_config_1 != ols_config_2

    # compare OLSConfig with other object
    assert ols_config_1 != "foo"
    assert ols_config_2 != {}


def test_config():
    """Test the Config model of the Global service configuration."""
    config = Config(
        {
            "llm_providers": [
                {
                    "name": "test_provider_name",
                    "type": "bam",
                    "url": "test_provider_url",
                    "credentials_path": "tests/config/secret/apitoken",
                    "models": [
                        {
                            "name": "test_model_name",
                            "url": "http://test_model_url/",
                            "credentials_path": "tests/config/secret/apitoken",
                        }
                    ],
                },
                {
                    "name": "rhelai_provider_name",
                    "type": "rhelai_vllm",
                    "url": "test_provider_url",
                    "credentials_path": "tests/config/secret/apitoken",
                    "models": [
                        {
                            "name": "test_model_name",
                            "url": "http://test_model_url/",
                            "credentials_path": "tests/config/secret/apitoken",
                        }
                    ],
                },
                {
                    "name": "rhoai_provider_name",
                    "type": "rhoai_vllm",
                    "url": "test_provider_url",
                    "credentials_path": "tests/config/secret/apitoken",
                    "models": [
                        {
                            "name": "test_model_name",
                            "url": "http://test_model_url/",
                            "credentials_path": "tests/config/secret/apitoken",
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
                "query_validation_method": "disabled",
                "certificate_directory": "/foo/bar/baz",
                "authentication_config": {"module": "foo"},
                "expire_llm_is_ready_persistent_state": 2,
            },
            "dev_config": {"disable_tls": "true"},
        }
    )
    assert len(config.llm_providers.providers) == 3
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
        str(
            config.llm_providers.providers["test_provider_name"]
            .models["test_model_name"]
            .url
        )
        == "http://test_model_url/"
    )
    assert (
        config.llm_providers.providers["test_provider_name"]
        .models["test_model_name"]
        .credentials
        == "secret_key"
    )
    assert (
        config.llm_providers.providers["rhoai_provider_name"].certificates_store
        == "/foo/bar/baz/ols.pem"
    )
    assert (
        config.llm_providers.providers["rhelai_provider_name"].certificates_store
        == "/foo/bar/baz/ols.pem"
    )
    assert (
        config.llm_providers.providers["test_provider_name"].certificates_store is None
    )

    assert config.ols_config.default_provider == "test_default_provider"
    assert config.ols_config.default_model == "test_default_model"
    assert config.ols_config.conversation_cache.type == "memory"
    assert config.ols_config.conversation_cache.memory.max_entries == 100
    assert config.ols_config.logging_config.app_log_level == logging.ERROR
    assert (
        config.ols_config.query_validation_method
        == constants.QueryValidationMethod.DISABLED
    )
    assert config.user_data_collector_config == UserDataCollectorConfig()
    assert config.ols_config.certificate_directory == "/foo/bar/baz"
    assert config.ols_config.system_prompt_path is None
    assert config.ols_config.system_prompt is None
    assert config.ols_config.authentication_config is not None
    assert config.ols_config.authentication_config.module is not None
    assert config.ols_config.authentication_config.module == "foo"
    assert config.ols_config.expire_llm_is_ready_persistent_state == 2


def test_config_equality():
    """Check the equality operator of Config class."""
    config = Config(
        {
            "llm_providers": [
                {
                    "name": "test_provider_name",
                    "type": "bam",
                    "url": "test_provider_url",
                    "credentials_path": "tests/config/secret/apitoken",
                    "models": [
                        {
                            "name": "test_model_name",
                            "url": "http://test_model_url/",
                            "credentials_path": "tests/config/secret/apitoken",
                        }
                    ],
                },
                {
                    "name": "rhelai_provider_name",
                    "type": "rhelai_vllm",
                    "url": "test_provider_url",
                    "credentials_path": "tests/config/secret/apitoken",
                    "models": [
                        {
                            "name": "test_model_name",
                            "url": "http://test_model_url/",
                            "credentials_path": "tests/config/secret/apitoken",
                        }
                    ],
                },
                {
                    "name": "rhoai_provider_name",
                    "type": "rhoai_vllm",
                    "url": "test_provider_url",
                    "credentials_path": "tests/config/secret/apitoken",
                    "models": [
                        {
                            "name": "test_model_name",
                            "url": "http://test_model_url/",
                            "credentials_path": "tests/config/secret/apitoken",
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
                "query_validation_method": "disabled",
                "certificate_directory": "/foo/bar/baz",
                "authentication_config": {"module": "foo"},
                "expire_llm_is_ready_persistent_state": 2,
            },
            "dev_config": {"disable_tls": "true"},
        }
    )
    assert config != "foo"
    assert config == config

    config2 = copy.deepcopy(config)
    assert config == config2

    config2.ols_config.expire_llm_is_ready_persistent_state = 3
    assert config != config2


def test_config_default_certificate_directory():
    """Test the Config model of the Global service configuration."""
    config = Config(
        {
            "llm_providers": [
                {
                    "name": "test_provider_name",
                    "type": "bam",
                    "url": "test_provider_url",
                    "credentials_path": "tests/config/secret/apitoken",
                    "models": [
                        {
                            "name": "test_model_name",
                            "url": "http://test_model_url/",
                            "credentials_path": "tests/config/secret/apitoken",
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
                "query_validation_method": "disabled",
            },
            "dev_config": {"disable_tls": "true"},
        }
    )
    assert (
        config.ols_config.certificate_directory
        == constants.DEFAULT_CERTIFICATE_DIRECTORY
    )


def test_config_improper_missing_model():
    """Test the Config model of the Global service configuration when model is missing."""
    with pytest.raises(InvalidConfigurationError, match="default_model is missing"):
        Config(
            {
                "llm_providers": [
                    {
                        "name": "test_provider_name",
                        "type": "bam",
                        "url": "http://test_provider_url",
                        "credentials_path": "tests/config/secret/apitoken",
                        "models": [
                            {
                                "name": "test_model_name",
                                "url": "http://test_model_url",
                                "credentials_path": "tests/config/secret/apitoken",
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
                "dev_config": {"disable_tls": "true"},
            }
        ).validate_yaml()


def test_config_improper_missing_provider():
    """Test the Config model of the Global service configuration when provider is missing."""
    with pytest.raises(InvalidConfigurationError, match="default_provider is missing"):
        Config(
            {
                "llm_providers": [
                    {
                        "name": "test_provider_name",
                        "type": "bam",
                        "url": "http://test_provider_url",
                        "credentials_path": "tests/config/secret/apitoken",
                        "models": [
                            {
                                "name": "test_model_name",
                                "url": "http://test_model_url",
                                "credentials_path": "tests/config/secret/apitoken",
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
                    "default_model": "test_default_model",
                },
                "dev_config": {"disable_tls": "true"},
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
                "dev_config": {"disable_tls": "true"},
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
                        "type": "bam",
                        "url": "http://test_provider_url",
                        "credentials_path": "tests/config/secret/apitoken",
                        "models": [
                            {
                                "name": "test_model_name",
                                "url": "http://test_model_url",
                                "credentials_path": "tests/config/secret/apitoken",
                            }
                        ],
                    }
                ],
                "ols_config": {
                    "default_provider": "test_provider_name",
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
                "dev_config": {"disable_tls": "true"},
            }
        ).validate_yaml()


def test_ols_config_with_invalid_validation_method():
    """Test the Ols config with invalid validation method."""
    ols_config = {
        "conversation_cache": {
            "type": "memory",
            "memory": {
                "max_entries": 100,
            },
        },
        "query_validation_method": False,
    }

    with pytest.raises(
        InvalidConfigurationError,
        match="Invalid query validation method",
    ):
        OLSConfig(ols_config).validate_yaml(True)


def test_logging_config_equality():
    """Test the LoggingConfig equality check."""
    logging_config_1 = LoggingConfig()
    logging_config_2 = LoggingConfig()

    # compare the same logging configs
    assert logging_config_1 == logging_config_2

    # compare different logging configs
    logging_config_2.app_log_level = 42
    assert logging_config_1 != logging_config_2

    # compare with value of different type
    other_value = "foo"
    assert logging_config_1 != other_value


def test_reference_content_index_constructor():
    """Test the ReferenceContentIndex constructor."""
    reference_content_index = ReferenceContentIndex(
        {
            "product_docs_index_id": "id",
            "product_docs_index_path": "/path/",
        }
    )
    assert reference_content_index.product_docs_index_id == "id"
    assert reference_content_index.product_docs_index_path == "/path/"


def test_reference_content_index_equality():
    """Test the ReferenceContentIndex equality check."""
    reference_content_index_1 = ReferenceContentIndex()
    reference_content_index_2 = ReferenceContentIndex()

    # compare the same configs
    assert reference_content_index_1 == reference_content_index_2

    # compare different configs
    reference_content_index_2.product_docs_index_id = "id"
    assert reference_content_index_1 != reference_content_index_2

    reference_content_index_2 = ReferenceContentIndex()
    reference_content_index_2.product_docs_index_path = "/path/"
    assert reference_content_index_1 != reference_content_index_2

    # compare with value of different type
    other_value = "foo"
    assert reference_content_index_1 != other_value


def test_reference_content_index_yaml_validation():
    """Test the ReferenceContentIndex YAML validation method."""
    reference_content_index = ReferenceContentIndex()
    # should not raise an exception
    reference_content_index.validate_yaml()

    # existing docs index path with set up product ID
    reference_content_index.product_docs_index_path = "."
    reference_content_index.product_docs_index_id = "foo"
    reference_content_index.validate_yaml()

    # existing docs index path, but no product ID
    reference_content_index.product_docs_index_path = "."
    reference_content_index.product_docs_index_id = None
    with pytest.raises(InvalidConfigurationError):
        reference_content_index.validate_yaml()

    # non-existing docs index path
    reference_content_index.product_docs_index_path = "foo"
    with pytest.raises(InvalidConfigurationError):
        reference_content_index.validate_yaml()

    # docs index does not point to a proper directory
    # but to special file
    reference_content_index.product_docs_index_path = "/dev/null"
    with pytest.raises(InvalidConfigurationError):
        reference_content_index.validate_yaml()

    # docs index point to a proper directory, that is not
    # readable by the service
    reference_content_index.product_docs_index_path = "/root"
    with pytest.raises(InvalidConfigurationError):
        reference_content_index.validate_yaml()


def test_reference_content_constructor():
    """Test the ReferenceContent constructor."""
    reference_content = ReferenceContent(
        {
            "embeddings_model_path": "/path/2/",
            "indexes": [
                {
                    "product_docs_index_id": "id",
                    "product_docs_index_path": "/path/1/",
                },
            ],
        }
    )
    assert reference_content.indexes[0] == ReferenceContentIndex(
        {"product_docs_index_id": "id", "product_docs_index_path": "/path/1/"}
    )
    assert reference_content.embeddings_model_path == "/path/2/"


def test_reference_content_equality():
    """Test the ReferenceContent equality check."""
    reference_content_1 = ReferenceContent()
    reference_content_2 = ReferenceContent()

    # compare the same configs
    assert reference_content_1 == reference_content_2

    # compare different configs
    reference_content_2.embeddings_model_path = "foo"
    assert reference_content_1 != reference_content_2

    reference_content_2 = ReferenceContent()
    reference_content_2.indexes = [
        ReferenceContentIndex(
            {"product_docs_index_id": "foo", "product_docs_index_path": "."},
        ),
    ]
    assert reference_content_1 != reference_content_2

    # compare with value of different type
    other_value = "foo"
    assert reference_content_1 != other_value


def test_reference_content_yaml_validation():
    """Test the ReferenceContent YAML validation method."""
    reference_content = ReferenceContent()
    # should not raise an exception
    reference_content.validate_yaml()

    # valid indexes
    reference_content.indexes = [
        ReferenceContentIndex(
            {"product_docs_index_id": "foo", "product_docs_index_path": "."}
        )
    ]
    reference_content.validate_yaml()

    # invalid indexes
    reference_content.indexes = [
        ReferenceContentIndex(
            {"product_docs_index_id": "foo", "product_docs_index_path": "/dev/null"}
        )
    ]
    with pytest.raises(InvalidConfigurationError):
        reference_content.validate_yaml()


def test_config_no_query_filter_node():
    """Test the Config model when query filter is not set at all."""
    config = Config(
        {
            "llm_providers": [
                {
                    "name": "openai",
                    "url": "http://test_provider_url",
                    "credentials_path": "tests/config/secret/apitoken",
                    "models": [
                        {
                            "name": "test_model_name",
                            "url": "http://test_model_url",
                            "credentials_path": "tests/config/secret/apitoken",
                        }
                    ],
                }
            ],
            "ols_config": {
                "default_provider": "openai",
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
            "dev_config": {
                "disable_tls": "true",
            },
        }
    )
    assert config.ols_config.query_filters is None


def test_config_no_query_filter():
    """Test the Config model when query filter list is empty."""
    config = Config(
        {
            "llm_providers": [
                {
                    "name": "openai",
                    "url": "http://test_provider_url",
                    "credentials_path": "tests/config/secret/apitoken",
                    "models": [
                        {
                            "name": "test_model_name",
                            "url": "http://test_model_url",
                            "credentials_path": "tests/config/secret/apitoken",
                        }
                    ],
                }
            ],
            "ols_config": {
                "default_provider": "openai",
                "default_model": "test_default_model",
                "conversation_cache": {
                    "type": "memory",
                    "memory": {
                        "max_entries": 100,
                    },
                },
                "query_filters": [],
                "logging_config": {
                    "app_log_level": "error",
                },
            },
            "dev_config": {
                "disable_tls": "true",
            },
        }
    )
    assert len(config.ols_config.query_filters) == 0


def test_config_improper_query_filter():
    """Test the Config model with improper query filter (no name) is set."""
    with pytest.raises(
        InvalidConfigurationError,
        match="name, pattern and replace_with need to be specified",
    ):
        Config(
            {
                "llm_providers": [
                    {
                        "name": "openai",
                        "url": "http://test_provider_url",
                        "credentials_path": "tests/config/secret/apitoken",
                        "models": [
                            {
                                "name": "test_model_name",
                                "url": "http://test_model_url",
                                "credentials_path": "tests/config/secret/apitoken",
                            }
                        ],
                    }
                ],
                "ols_config": {
                    "default_provider": "openai",
                    "default_model": "test_default_model",
                    "query_filters": [
                        {
                            "pattern": "test_regular_expression",
                            "replace_with": "test_replace_with",
                        }
                    ],
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
                "dev_config": {
                    "disable_tls": "true",
                },
            }
        ).validate_yaml()


def test_config_with_multiple_query_filter():
    """Test the Config model with multiple query filter is set."""
    config = Config(
        {
            "llm_providers": [
                {
                    "name": "openai",
                    "url": "http://test_provider_url",
                    "credentials_path": "tests/config/secret/apitoken",
                    "models": [
                        {
                            "name": "test_model_name",
                            "url": "http://test.io",
                            "credentials_path": "tests/config/secret/apitoken",
                        }
                    ],
                },
            ],
            "ols_config": {
                "default_provider": "openai",
                "default_model": "test_model_name",
                "conversation_cache": {
                    "type": "memory",
                    "memory": {
                        "max_entries": 100,
                    },
                },
                "logging_config": {
                    "app_log_level": "error",
                },
                "query_filters": [
                    {
                        "name": "filter1",
                        "pattern": r"(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
                        "replace_with": "redacted",
                    },
                    {
                        "name": "filter2",
                        "pattern": r"(?:https?://)?(?:www\.)?[\w\.-]+\.\w+",
                        "replace_with": "",
                    },
                ],
            },
            "dev_config": {
                "disable_tls": "true",
            },
        }
    )
    assert config.ols_config.query_filters[0].name == "filter1"
    assert (
        config.ols_config.query_filters[0].pattern
        == r"(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
    )
    assert config.ols_config.query_filters[0].replace_with == "redacted"
    assert config.ols_config.query_filters[1].name == "filter2"
    assert (
        config.ols_config.query_filters[1].pattern
        == r"(?:https?://)?(?:www\.)?[\w\.-]+\.\w+"
    )
    assert config.ols_config.query_filters[1].replace_with == ""


def test_config_invalid_regex_query_filter():
    """Test the Config model with invalid query filter pattern."""
    with pytest.raises(
        InvalidConfigurationError,
        match="pattern is invalid",
    ):
        Config(
            {
                "llm_providers": [
                    {
                        "name": "openai",
                        "url": "http://test_provider_url",
                        "credentials_path": "tests/config/secret/apitoken",
                        "models": [
                            {
                                "name": "test_model_name",
                                "url": "http://test_model_url",
                                "credentials_path": "tests/config/secret/apitoken",
                            }
                        ],
                    }
                ],
                "ols_config": {
                    "default_provider": "openai",
                    "default_model": "test_default_model",
                    "query_filters": [
                        {
                            "name": "test_name",
                            "pattern": "[",
                            "replace_with": "test_replace_with",
                        }
                    ],
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
                "dev_config": {
                    "disable_tls": "true",
                },
            }
        ).validate_yaml()


def test_query_filter_constructor():
    """Test checks made by QueryFilter constructor."""
    # no input
    query_filter = QueryFilter(None)
    assert query_filter.name is None
    assert query_filter.pattern is None
    assert query_filter.replace_with is None

    # proper input
    query_filter = QueryFilter(
        {"name": "NAME", "pattern": "PATTERN", "replace_with": "REPLACE_WITH"}
    )
    assert query_filter.name == "NAME"
    assert query_filter.pattern == "PATTERN"
    assert query_filter.replace_with == "REPLACE_WITH"

    # missing inputs
    with pytest.raises(
        InvalidConfigurationError,
        match="name, pattern and replace_with need to be specified",
    ):
        QueryFilter({"pattern": "PATTERN", "replace_with": "REPLACE_WITH"})
    with pytest.raises(
        InvalidConfigurationError,
        match="name, pattern and replace_with need to be specified",
    ):
        QueryFilter({"name": "NAME", "replace_with": "REPLACE_WITH"})
    with pytest.raises(
        InvalidConfigurationError,
        match="name, pattern and replace_with need to be specified",
    ):
        QueryFilter({"name": "NAME", "pattern": "PATTERN"})


def test_query_filter_validation():
    """Test method to validate query filter settings."""

    def get_query_filter():
        """Construct new fully-configured filter from scratch."""
        return QueryFilter(
            {"name": "NAME", "pattern": "PATTERN", "replace_with": "REPLACE_WITH"}
        )

    query_filter = get_query_filter()
    query_filter.name = None
    with pytest.raises(InvalidConfigurationError, match="name is missing"):
        query_filter.validate_yaml()

    query_filter = get_query_filter()
    query_filter.pattern = None
    with pytest.raises(InvalidConfigurationError, match="pattern is missing"):
        query_filter.validate_yaml()

    query_filter = get_query_filter()
    query_filter.replace_with = None
    with pytest.raises(InvalidConfigurationError, match="replace_with is missing"):
        query_filter.validate_yaml()


def get_query_filters():
    """Construct two instances of QueryFilter class."""
    query_filter_1 = QueryFilter()
    query_filter_2 = QueryFilter()
    return query_filter_1, query_filter_2


def test_query_filter_equality(subtests):
    """Test the QueryFilter equality check."""
    # compare two objects with the same content
    query_filter_1, query_filter_2 = get_query_filters()
    assert query_filter_1 == query_filter_2

    # compare with value of different type
    other_value = "foo"
    assert query_filter_1 != other_value

    # compare two objects with different content
    with subtests.test(msg="Different attribute: name"):
        query_filter_1, query_filter_2 = get_query_filters()
        query_filter_1.name = "foo"
        query_filter_2.name = "bar"
        assert query_filter_1 != query_filter_2

    # compare two objects with different content
    with subtests.test(msg="Different attribute: pattern"):
        query_filter_1, query_filter_2 = get_query_filters()
        query_filter_1.pattern = "foo.*"
        query_filter_2.pattern = "bar.*"
        assert query_filter_1 != query_filter_2

    # compare two objects with different content
    with subtests.test(msg="Different attribute: replace_with"):
        query_filter_1, query_filter_2 = get_query_filters()
        query_filter_1.replace_with = "-foo-"
        query_filter_2.replace_with = "-bar-"
        assert query_filter_1 != query_filter_2


def test_query_filter_equality_null_values(subtests):
    """Test the QueryFilter equality check."""
    # compare two objects with different content
    with subtests.test(msg="Different attribute: name"):
        query_filter_1, query_filter_2 = get_query_filters()
        query_filter_1.name = None
        query_filter_2.name = "bar"
        assert query_filter_1 != query_filter_2

    # compare two objects with different content
    with subtests.test(msg="Different attribute: pattern"):
        query_filter_1, query_filter_2 = get_query_filters()
        query_filter_1.pattern = None
        query_filter_2.pattern = "bar.*"
        assert query_filter_1 != query_filter_2

    # compare two objects with different content
    with subtests.test(msg="Different attribute: replace_with"):
        query_filter_1, query_filter_2 = get_query_filters()
        query_filter_1.replace_with = None
        query_filter_2.replace_with = "-bar-"
        assert query_filter_1 != query_filter_2


def test_authentication_config_validation_proper_config():
    """Test method to validate authentication config."""
    # default module
    AuthenticationConfig(
        skip_tls_verification=True,
        k8s_cluster_api="http://cluster.org/foo",
        k8s_ca_cert_path="tests/config/empty_cert.crt",
    )
    # custom module
    AuthenticationConfig(
        module=constants.DEFAULT_AUTHENTICATION_MODULE,
        skip_tls_verification=True,
        k8s_cluster_api="http://cluster.org/foo",
        k8s_ca_cert_path="tests/config/empty_cert.crt",
    )


def test_authentication_config_wrong_module():
    """Test method to validate authentication config when wrong module is specified."""
    #  no module
    cfg = AuthenticationConfig(
        module=None,
        skip_tls_verification=True,
        k8s_cluster_api="http://cluster.org/foo",
        k8s_ca_cert_path="tests/config/empty_cert.crt",
    )
    with pytest.raises(
        InvalidConfigurationError, match="Authentication module is not setup"
    ):
        cfg.validate_yaml()

    # empty module
    cfg = AuthenticationConfig(
        module="",
        skip_tls_verification=True,
        k8s_cluster_api="http://cluster.org/foo",
        k8s_ca_cert_path="tests/config/empty_cert.crt",
    )
    with pytest.raises(
        InvalidConfigurationError, match="invalid authentication module"
    ):
        cfg.validate_yaml()

    # custom module
    cfg = AuthenticationConfig(
        module="foo",
        skip_tls_verification=True,
        k8s_cluster_api="http://cluster.org/foo",
        k8s_ca_cert_path="tests/config/empty_cert.crt",
    )
    with pytest.raises(
        InvalidConfigurationError, match="invalid authentication module"
    ):
        cfg.validate_yaml()


def test_authentication_config_k8s_cluster_api():
    """Test method to validate authentication config."""
    # k8s_cluster_api is optional
    AuthenticationConfig(
        skip_tls_verification=True,
    )

    # but when provided, it needs to be valid URL pattern
    AuthenticationConfig(
        skip_tls_verification=True,
        k8s_cluster_api="http://cluster.org/foo",
    )

    with pytest.raises(
        ValidationError,
        match="Input should be a valid URL, input is empty",
    ):
        AuthenticationConfig(
            skip_tls_verification=True,
            k8s_cluster_api="",
        )

    with pytest.raises(
        ValidationError,
        match="Input should be a valid URL, relative URL without a base",
    ):
        AuthenticationConfig(
            skip_tls_verification=True,
            k8s_cluster_api="this-is-not-valid-url",
        )

    with pytest.raises(
        ValidationError,
        match="Input should be a valid URL, relative URL without a base",
    ):
        AuthenticationConfig(
            skip_tls_verification=True,
            k8s_cluster_api="None",
        )


def test_authentication_config_validation_k8s_ca_cert_path():
    """Test method to validate authentication config when cert path is empty."""
    # k8s_ca_cert_path is optional
    AuthenticationConfig(
        skip_tls_verification=True,
    )

    # but when provided, it needs to be a valid file path
    AuthenticationConfig(
        skip_tls_verification=True,
        k8s_ca_cert_path=__file__,  # just use some existing file path here
    )

    with pytest.raises(ValidationError, match="Path does not point to a file"):
        AuthenticationConfig(
            skip_tls_verification=True,
            k8s_ca_cert_path="",
        )

    with pytest.raises(ValidationError, match="Path does not point to a file"):
        AuthenticationConfig(
            skip_tls_verification=True,
            k8s_ca_cert_path="/dev/null/foo",  # that file can not exists
        )
    with pytest.raises(ValidationError, match="Path does not point to a file"):
        AuthenticationConfig(
            skip_tls_verification=True,
            k8s_ca_cert_path="/dev/null",
        )

    with pytest.raises(ValidationError, match="Path does not point to a file"):
        AuthenticationConfig(
            skip_tls_verification=True,
            k8s_ca_cert_path="None",
        )


def test_user_data_config__feedback(tmpdir):
    """Tests the UserDataCollection model, feedback part."""
    # valid configuration
    user_data = UserDataCollection(
        feedback_disabled=False, feedback_storage=tmpdir.strpath
    )
    assert user_data.feedback_disabled is False
    assert user_data.feedback_storage == tmpdir.strpath

    # enabled needs feedback_storage
    with pytest.raises(
        ValueError,
        match="feedback_storage is required when feedback is enabled",
    ):
        UserDataCollection(feedback_disabled=False)

    # disabled doesn't need feedback_storage
    user_data = UserDataCollection(feedback_disabled=True)
    assert user_data.feedback_disabled is True
    assert user_data.feedback_storage is None


def test_user_data_config__transcripts(tmpdir):
    """Tests the UserDataCollection model, transripts part."""
    # valid configuration
    user_data = UserDataCollection(
        transcripts_disabled=False, transcripts_storage=tmpdir.strpath
    )
    assert user_data.transcripts_disabled is False
    assert user_data.transcripts_storage == tmpdir.strpath

    # enabled needs transcripts_storage
    with pytest.raises(
        ValueError,
        match="transcripts_storage is required when transcripts capturing is enabled",
    ):
        UserDataCollection(transcripts_disabled=False)

    # disabled doesn't need transcripts_storage
    user_data = UserDataCollection(transcripts_disabled=True)
    assert user_data.transcripts_disabled is True
    assert user_data.transcripts_storage is None


def test_dev_config_defaults():
    """Test the DevConfig model with default values."""
    dev_config = DevConfig()
    assert dev_config.pyroscope_url is None
    assert dev_config.enable_dev_ui is False
    assert dev_config.llm_params == {}
    assert dev_config.disable_auth is False
    assert dev_config.disable_tls is False
    assert dev_config.k8s_auth_token is None
    assert dev_config.run_on_localhost is False
    assert dev_config.enable_system_prompt_override is False
    assert dev_config.uvicorn_port_number is None


def get_dev_configs():
    """Construct two instances of DevConfig class."""
    dev_config_1 = DevConfig()
    dev_config_2 = DevConfig()
    return dev_config_1, dev_config_2


def test_dev_config_equality(subtests):
    """Test the DevConfig equality check."""
    # compare two objects with the same content
    dev_config_1, dev_config_2 = get_dev_configs()
    assert dev_config_1 == dev_config_2

    # compare with value of different type
    other_value = "foo"
    assert dev_config_1 != other_value

    # compare two objects with different content
    with subtests.test(msg="Different attritubte: enable_dev_ui"):
        dev_config_1, dev_config_2 = get_dev_configs()
        dev_config_1.enable_dev_ui = True
        dev_config_2.enable_dev_ui = False
        assert dev_config_1 != dev_config_2

    with subtests.test(msg="Different attritubte: llm_params"):
        dev_config_1, dev_config_2 = get_dev_configs()
        dev_config_1.llm_params = {}
        dev_config_2.llm_params = {"foo": "bar"}
        assert dev_config_1 != dev_config_2

    with subtests.test(msg="Different attritubte: disable_auth"):
        dev_config_1, dev_config_2 = get_dev_configs()
        dev_config_1.disable_auth = True
        dev_config_2.disable_auth = False
        assert dev_config_1 != dev_config_2

    with subtests.test(msg="Different attritubte: disable_tls"):
        dev_config_1, dev_config_2 = get_dev_configs()
        dev_config_1.disable_tls = True
        dev_config_2.disable_tls = False
        assert dev_config_1 != dev_config_2

    with subtests.test(msg="Different attritubte: enable_system_prompt_override"):
        dev_config_1, dev_config_2 = get_dev_configs()
        dev_config_1.enable_system_prompt_override = True
        dev_config_2.enable_system_prompt_override = False
        assert dev_config_1 != dev_config_2

    with subtests.test(msg="Different attritubte: run_on_localhost"):
        dev_config_1, dev_config_2 = get_dev_configs()
        dev_config_1.run_on_localhost = True
        dev_config_2.run_on_localhost = False
        assert dev_config_1 != dev_config_2

    with subtests.test(msg="Different attritubte: pyroscope_url"):
        dev_config_1, dev_config_2 = get_dev_configs()
        dev_config_1.pyroscope_url = None
        dev_config_2.pyroscope_url = "http://test.com"
        assert dev_config_1 != dev_config_2

    with subtests.test(msg="Different attritubte: k8s_auth_token"):
        dev_config_1, dev_config_2 = get_dev_configs()
        dev_config_1.k8s_auth_token = None
        dev_config_2.k8s_auth_token = "***token***"  # noqa: S105
        assert dev_config_1 != dev_config_2


def test_dev_config_bool_inputs():
    """Test the DevConfig model with boolean inputs."""
    true_values = {"1", "on", "t", "true", "y", "yes"}
    false_values = {"0", "off", "f", "false", "n", "no"}

    for value in true_values:
        dev_config = DevConfig(enable_dev_ui=value)
        assert dev_config.enable_dev_ui is True

    for value in false_values:
        dev_config = DevConfig(enable_dev_ui=value)
        assert dev_config.enable_dev_ui is False


def test_user_data_collection_config__defaults():
    """Test the UserDataCollection model with default values."""
    udc_config = UserDataCollectorConfig()
    assert udc_config.data_storage is None
    assert udc_config.log_level == logging.INFO
    assert udc_config.collection_interval == 2 * 60 * 60
    assert udc_config.run_without_initial_wait is False
    assert udc_config.ingress_env == "prod"
    assert udc_config.cp_offline_token is None


def test_user_data_collection_config__logging_level():
    """Test the UserDataCollection model with logging level."""
    udc_config = UserDataCollectorConfig(log_level="debug")
    assert udc_config.log_level == logging.DEBUG

    udc_config = UserDataCollectorConfig(log_level="DEBUG")
    assert udc_config.log_level == logging.DEBUG


def test_user_data_collection_config__token_expectation():
    """Test the UserDataCollection model with token expectation."""
    udc_config = UserDataCollectorConfig(
        ingress_env="stage",
        cp_offline_token="123",  # noqa: S106
    )
    assert udc_config.ingress_env == "stage"
    assert udc_config.cp_offline_token == "123"  # noqa: S105

    with pytest.raises(
        ValueError,
        match="cp_offline_token is required in stage environment",
    ):
        UserDataCollectorConfig(ingress_env="stage", cp_offline_token=None)


def test_ols_config_with_system_prompt(tmpdir):
    """Test the OLSConfig model with system prompt."""
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
            "system_prompt_path": "tests/config/system_prompt.txt",
        }
    )
    assert ols_config.default_provider == "test_default_provider"
    assert ols_config.default_model == "test_default_model"
    assert ols_config.conversation_cache.type == "memory"
    assert ols_config.conversation_cache.memory.max_entries == 100
    assert ols_config.logging_config.app_log_level == logging.INFO
    assert (
        ols_config.query_validation_method == constants.QueryValidationMethod.DISABLED
    )
    assert ols_config.user_data_collection == UserDataCollection()
    assert ols_config.reference_content is None
    assert ols_config.authentication_config == AuthenticationConfig(
        module=constants.DEFAULT_AUTHENTICATION_MODULE
    )
    assert ols_config.extra_ca == []
    assert ols_config.certificate_directory == constants.DEFAULT_CERTIFICATE_DIRECTORY
    assert ols_config.system_prompt_path is None
    assert ols_config.system_prompt == "This is test system prompt!"


def test_ols_config_with_non_existing_system_prompt(tmpdir):
    """Test the OLSConfig model with system prompt path specification that does not exist."""
    with pytest.raises(
        FileNotFoundError,
        match="No such file or directory: 'tests/config/non_existing_file.txt'",
    ):
        OLSConfig(
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
                "system_prompt_path": "tests/config/non_existing_file.txt",
            }
        )


def test_ols_config_with_non_readable_system_prompt(tmpdir):
    """Test the OLSConfig model with system prompt path specification that is not readable."""
    with pytest.raises(
        IsADirectoryError,
        match="Is a directory: 'tests/config/'",
    ):
        OLSConfig(
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
                "system_prompt_path": "tests/config/",
            }
        )


def test_ols_config_with_quota_handlers_section():
    """Test OLSConfig model with quota handlers section specified."""
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
            "quota_handlers": {
                "storage": {
                    "host": "",
                    "port": 5432,
                    "dbname": "test",
                    "user": "tester",
                    "password_path": "tests/config/postgres_password.txt",
                    "ssl_mode": "disable",
                },
                "limiters": [
                    {
                        "name": "user_monthly_limits",
                        "type": "user_limiter",
                        "initial_quota": 1000,
                        "quota_increase": 10,
                        "period": "5 minutes",
                    },
                    {
                        "name": "cluster_monthly_limits",
                        "type": "cluster_limiter",
                        "initial_quota": 2000,
                        "quota_increase": 100,
                        "period": "5 minutes",
                    },
                ],
                "scheduler": {
                    "period": 100,
                },
            },
        }
    )
    assert ols_config.quota_handlers is not None
    assert ols_config.quota_handlers.scheduler is not None
    assert ols_config.quota_handlers.storage is not None
    assert ols_config.quota_handlers.limiters is not None


def test_ols_config_with_quota_handlers_section_without_storage():
    """Test OLSConfig model with quota handlers section specified but w/o storage part."""
    with pytest.raises(
        InvalidConfigurationError,
        match="Missing storage configuration for quota limiters",
    ):
        OLSConfig(
            {
                "default_provider": "test_default_provider",
                "default_model": "test_default_model",
                "conversation_cache": {
                    "type": "memory",
                    "memory": {
                        "max_entries": 100,
                    },
                },
                "quota_handlers": {
                    "limiters": [
                        {
                            "name": "user_monthly_limits",
                            "type": "user_limiter",
                            "initial_quota": 1000,
                            "quota_increase": 10,
                            "period": "5 minutes",
                        },
                        {
                            "name": "cluster_monthly_limits",
                            "type": "cluster_limiter",
                            "initial_quota": 2000,
                            "quota_increase": 100,
                            "period": "5 minutes",
                        },
                    ],
                    "scheduler": {
                        "period": 100,
                    },
                },
            }
        )


def test_ols_config_with_quota_handlers_section_without_scheduler():
    """Test OLSConfig model with quota handlers section specified but w/o scheduler part."""
    with pytest.raises(
        InvalidConfigurationError,
        match="Missing scheduler configuration for quota limiters",
    ):
        OLSConfig(
            {
                "default_provider": "test_default_provider",
                "default_model": "test_default_model",
                "conversation_cache": {
                    "type": "memory",
                    "memory": {
                        "max_entries": 100,
                    },
                },
                "quota_handlers": {
                    "storage": {
                        "host": "",
                        "port": 5432,
                        "dbname": "test",
                        "user": "tester",
                        "password_path": "postgres_password.txt",
                        "ssl_mode": "disable",
                    },
                    "limiters": [
                        {
                            "name": "user_monthly_limits",
                            "type": "user_limiter",
                            "initial_quota": 1000,
                            "quota_increase": 10,
                            "period": "5 minutes",
                        },
                        {
                            "name": "cluster_monthly_limits",
                            "type": "cluster_limiter",
                            "initial_quota": 2000,
                            "quota_increase": 100,
                            "period": "5 minutes",
                        },
                    ],
                },
            }
        )


def test_ols_config_with_quota_handlers_section_without_limiters():
    """Test OLSConfig model with quota handlers section specified w/o limiters section."""
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
            "quota_handlers": {
                "storage": {
                    "host": "",
                    "port": 5432,
                    "dbname": "test",
                    "user": "tester",
                    "password_path": "tests/config/postgres_password.txt",
                    "ssl_mode": "disable",
                },
                "scheduler": {
                    "period": 100,
                },
            },
        }
    )
    assert ols_config.quota_handlers is not None
    assert ols_config.quota_handlers.scheduler is not None
    assert ols_config.quota_handlers.storage is not None
    assert ols_config.quota_handlers.limiters is not None


def test_ols_config_with_quota_handlers_section_empty_limiters():
    """Test OLSConfig model with quota handlers section specified with empty limiters section."""
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
            "quota_handlers": {
                "limiters": [],
                "storage": {
                    "host": "",
                    "port": 5432,
                    "dbname": "test",
                    "user": "tester",
                    "password_path": "tests/config/postgres_password.txt",
                    "ssl_mode": "disable",
                },
                "scheduler": {
                    "period": 100,
                },
            },
        }
    )
    assert ols_config.quota_handlers is not None
    assert ols_config.quota_handlers.scheduler is not None
    assert ols_config.quota_handlers.storage is not None
    assert ols_config.quota_handlers.limiters is not None


def test_ols_config_with_quota_handlesr_missing_name():
    """Test OLSConfig model with quota handlers section specified."""
    with pytest.raises(
        InvalidConfigurationError,
        match="limiter name is missing",
    ):
        OLSConfig(
            {
                "default_provider": "test_default_provider",
                "default_model": "test_default_model",
                "conversation_cache": {
                    "type": "memory",
                    "memory": {
                        "max_entries": 100,
                    },
                },
                "quota_handlers": {
                    "storage": {
                        "host": "",
                        "port": 5432,
                        "dbname": "test",
                        "user": "tester",
                        "password_path": "tests/config/postgres_password.txt",
                        "ssl_mode": "disable",
                    },
                    "limiters": [
                        {
                            "type": "user_limiter",
                            "initial_quota": 1000,
                            "quota_increase": 10,
                            "period": "5 minutes",
                        },
                        {
                            "type": "cluster_limiter",
                            "initial_quota": 2000,
                            "quota_increase": 100,
                            "period": "5 minutes",
                        },
                    ],
                    "scheduler": {
                        "period": 100,
                    },
                },
            }
        )
