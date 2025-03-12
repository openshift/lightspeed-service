"""Unit tests for Azure OpenAI provider."""

import time
from unittest.mock import patch

import httpx
import pytest
from azure.core.credentials import AccessToken
from langchain_openai import AzureChatOpenAI

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.azure_openai import AzureOpenAI, token_is_expired


@pytest.fixture
def provider_config():
    """Fixture with provider configuration for Azure OpenAI."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "azure_openai",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "deployment_name": "test_deployment_name",
            "models": [
                {
                    "name": "test_model_name",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_credentials_directory():
    """Fixture with provider configuration for Azure OpenAI."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "azure_openai",
            "url": "test_url",
            "credentials_path": "tests/config/secret",
            "deployment_name": "test_deployment_name",
            "models": [
                {
                    "name": "test_model_name",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_with_specific_parameters():
    """Fixture with provider configuration for Azure OpenAI with specific parameters."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "azure_openai",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "deployment_name": "test_deployment_name",
            "azure_openai_config": {
                "url": "http://azure.com",
                "deployment_name": "azure_deployment_name",
                "credentials_path": "tests/config/secret2/apitoken",
            },
            "models": [
                {
                    "name": "test_model_name",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_without_credentials():
    """Fixture with provider configuration for Azure OpenAI without credentials."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "azure_openai",
            "url": "test_url",
            "deployment_name": "test_deployment_name",
            "models": [
                {
                    "name": "test_model_name",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_without_tenant_id():
    """Fixture with provider configuration for Azure OpenAI without tenant_id."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "azure_openai",
            "url": "test_url",
            "deployment_name": "test_deployment_name",
            "azure_openai_config": {
                "url": "http://azure.com",
                "deployment_name": "azure_deployment_name",
                "credentials_path": "tests/config/secret_azure_no_tenant_id",
            },
            "models": [
                {
                    "name": "test_model_name",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_without_client_id():
    """Fixture with provider configuration for Azure OpenAI without client_id."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "azure_openai",
            "url": "test_url",
            "deployment_name": "test_deployment_name",
            "azure_openai_config": {
                "url": "http://azure.com",
                "deployment_name": "azure_deployment_name",
                "credentials_path": "tests/config/secret_azure_no_client_id",
            },
            "models": [
                {
                    "name": "test_model_name",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_without_client_secret():
    """Fixture with provider configuration for Azure OpenAI without client_secret."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "azure_openai",
            "url": "test_url",
            "deployment_name": "test_deployment_name",
            "azure_openai_config": {
                "url": "http://azure.com",
                "deployment_name": "azure_deployment_name",
                "credentials_path": "tests/config/secret_azure_no_client_secret",
            },
            "models": [
                {
                    "name": "test_model_name",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_access_token_related_parameters():
    """Fixture with provider configuration for Azure OpenAI with parameters to get access token."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "azure_openai",
            "url": "test_url",
            "deployment_name": "test_deployment_name",
            "azure_openai_config": {
                "url": "http://azure.com",
                "deployment_name": "azure_deployment_name",
                "credentials_path": "tests/config/secret_azure_tenant_id_client_id_client_secret",
            },
            "models": [
                {
                    "name": "test_model_name",
                }
            ],
        }
    )


def test_basic_interface(provider_config):
    """Test basic interface."""
    azure_openai = AzureOpenAI(
        model="uber-model", params={}, provider_config=provider_config
    )
    llm = azure_openai.load()
    assert isinstance(llm, AzureChatOpenAI)
    assert azure_openai.default_params

    # parameter presence test
    assert "model" in azure_openai.default_params
    assert "deployment_name" in azure_openai.default_params
    assert "api_key" in azure_openai.default_params
    assert "azure_endpoint" in azure_openai.default_params
    assert "max_tokens" in azure_openai.default_params
    assert "api_version" in azure_openai.default_params

    # test parameter values taken from config
    assert azure_openai.default_params["deployment_name"] == "test_deployment_name"

    # API key should be loaded from secret
    assert azure_openai.default_params["api_key"] == "secret_key"

    assert azure_openai.default_params["azure_endpoint"] == "test_url"

    # check the HTTP client parameter
    assert "http_client" in azure_openai.default_params
    assert azure_openai.default_params["http_client"] is not None

    client = azure_openai.default_params["http_client"]
    assert isinstance(client, httpx.Client)


def test_credentials_in_directory_handling(provider_config_credentials_directory):
    """Test that credentials in directory is handled as expected."""
    azure_openai = AzureOpenAI(
        model="uber-model",
        params={},
        provider_config=provider_config_credentials_directory,
    )
    llm = azure_openai.load()
    assert isinstance(llm, AzureChatOpenAI)
    assert azure_openai.default_params

    assert azure_openai.default_params["api_key"] == "secret_key"


def test_loading_provider_specific_parameters(provider_config_with_specific_parameters):
    """Test if provider-specific parameters are loaded too."""
    azure_openai = AzureOpenAI(
        model="uber-model",
        params={},
        provider_config=provider_config_with_specific_parameters,
    )
    llm = azure_openai.load()
    assert isinstance(llm, AzureChatOpenAI)
    assert azure_openai.default_params

    # parameter presence test
    assert "model" in azure_openai.default_params
    assert "deployment_name" in azure_openai.default_params
    assert "api_key" in azure_openai.default_params
    assert "azure_endpoint" in azure_openai.default_params
    assert "max_tokens" in azure_openai.default_params
    assert "api_version" in azure_openai.default_params

    # test parameter values taken from provider-specific config
    assert azure_openai.default_params["deployment_name"] == "azure_deployment_name"
    assert azure_openai.default_params["azure_endpoint"] == "http://azure.com/"

    # API key should be loaded from secret
    assert azure_openai.default_params["api_key"] == "secret_key_2"

    # parameters taken from provier-specific configuration
    # which takes precedence over regular configuration
    assert azure_openai.url == "http://azure.com/"
    assert azure_openai.credentials == "secret_key_2"


def test_params_handling(provider_config):
    """Test that not allowed parameters are removed before model init."""
    # first three parameters should be removed before model init
    # rest need to stay
    params = {
        "unknown_parameter": "foo",
        "min_new_tokens": 1,
        "max_new_tokens": 10,
        "temperature": 0.3,
        "verbose": True,
        "api_version": "2023-12-31",
    }

    azure_openai = AzureOpenAI(
        model="uber-model", params=params, provider_config=provider_config
    )
    llm = azure_openai.load()
    assert isinstance(llm, AzureChatOpenAI)
    assert azure_openai.default_params
    assert azure_openai.params

    # known parameters should be there
    assert "temperature" in azure_openai.params
    assert "verbose" in azure_openai.params
    assert "api_version" in azure_openai.params
    assert azure_openai.params["temperature"] == 0.3
    assert azure_openai.params["verbose"] is True
    assert azure_openai.params["api_version"] == "2023-12-31"

    # unknown parameters should be filtered out
    assert "min_new_tokens" not in azure_openai.params
    assert "max_new_tokens" not in azure_openai.params
    assert "unknown_parameter" not in azure_openai.params

    # taken from configuration
    assert azure_openai.url == "test_url"
    assert azure_openai.credentials == "secret_key"


def test_api_version_can_not_be_none(provider_config):
    """Test that api_version parameter can not be None."""
    params = {
        "api_version": None,
    }

    azure_openai = AzureOpenAI(
        model="uber-model", params=params, provider_config=provider_config
    )

    # api_version is required parameter and can not be None
    # in new OpenAI library it is checked internally!
    with pytest.raises(ValueError, match="api_version"):
        azure_openai.load()


def test_none_params_handling(provider_config):
    """Test that not allowed parameters are removed before model init."""
    # first three parameters should be removed before model init
    # rest need to stay
    params = {
        "unknown_parameter": None,
        "min_new_tokens": None,
        "max_new_tokens": None,
        "organization": None,
        "cache": None,
    }

    azure_openai = AzureOpenAI(
        model="uber-model", params=params, provider_config=provider_config
    )
    llm = azure_openai.load()
    assert isinstance(llm, AzureChatOpenAI)
    assert azure_openai.default_params
    assert azure_openai.params

    # known parameters should be there
    assert "organization" in azure_openai.params
    assert "cache" in azure_openai.params
    assert azure_openai.params["organization"] is None
    assert azure_openai.params["cache"] is None

    # unknown parameters should be filtered out
    assert "min_new_tokens" not in azure_openai.params
    assert "max_new_tokens" not in azure_openai.params
    assert "unknown_parameter" not in azure_openai.params


def test_missing_credentials_check(provider_config_without_credentials):
    """Test that check for missing credentials is in place ."""
    with pytest.raises(ValueError, match="Credentials for API token is not set"):
        AzureOpenAI(
            model="uber-model",
            params={},
            provider_config=provider_config_without_credentials,
        )


def test_missing_tenant_id(provider_config_without_tenant_id):
    """Test that check for missing tenant_id is in place ."""
    with pytest.raises(
        ValueError, match="tenant_id should be set in azure_openai_config"
    ):
        AzureOpenAI(
            model="uber-model",
            params={},
            provider_config=provider_config_without_tenant_id,
        )


def test_missing_client_id(provider_config_without_client_id):
    """Test that check for missing client_id is in place ."""
    with pytest.raises(
        ValueError, match="client_id should be set in azure_openai_config"
    ):
        AzureOpenAI(
            model="uber-model",
            params={},
            provider_config=provider_config_without_client_id,
        )


def test_missing_client_secret(provider_config_without_client_secret):
    """Test that check for missing credentials_path is in place ."""
    with pytest.raises(
        ValueError, match="client_secret should be set in azure_openai_config"
    ):
        AzureOpenAI(
            model="uber-model",
            params={},
            provider_config=provider_config_without_client_secret,
        )


class MockedAccessToken:
    """Mock class representing AccessToken that can be retrieved from Azure auth. mechanism."""

    def __init__(self):
        """Construct mocked access token class."""
        self.token = "this-is-access-token"  # noqa: S105
        self.expires_on = 3600  # random value


class MockedCredential:
    """Mock class representing Credential class that is used to retrieve access token."""

    def __init__(self, *args, **kwargs):
        """Construct mocked credential class."""

    def get_token(self, url):
        """Request an access token."""
        return MockedAccessToken()


class MockedCredentialThrowingException:
    """Mock class representing Credential class that is used to retrieve access token."""

    def __init__(self, *args, **kwargs):
        """Construct mocked credential class."""

    def get_token(self, url):
        """Request an access token."""
        raise Exception("Error getting token")


@pytest.fixture
def mocked_token_cache():
    """Fake token cache."""

    class MockedTokenCache:
        access_token = "cached_token"  # noqa: S105
        expires_on = 0

    return MockedTokenCache


@patch(
    "ols.src.llms.providers.azure_openai.ClientSecretCredential", new=MockedCredential
)
def test_retrieve_access_token(provider_config_access_token_related_parameters):
    """Test that access token is being retrieved."""
    azure_openai = AzureOpenAI(
        model="uber-model",
        params={},
        provider_config=provider_config_access_token_related_parameters,
    )
    assert "api_key" not in azure_openai.default_params
    assert (
        azure_openai.default_params["azure_ad_token"]
        == "this-is-access-token"  # noqa: S105
    )


@patch(
    "ols.src.llms.providers.azure_openai.ClientSecretCredential",
    new=MockedCredentialThrowingException,
)
@patch("ols.src.llms.providers.azure_openai.TokenCache.expires_on", new=0)
def test_retrieve_access_token_on_error(
    provider_config_access_token_related_parameters,
):
    """Test how error is handled during accessing token."""
    azure_openai = AzureOpenAI(
        model="uber-model",
        params={},
        provider_config=provider_config_access_token_related_parameters,
    )
    assert "api_key" not in azure_openai.default_params
    assert azure_openai.default_params["azure_ad_token"] is None


def test_token_is_expired():
    """Test token expiration check."""
    # expires_on is 0 - means it is not set
    with patch("ols.src.llms.providers.azure_openai.TokenCache.expires_on", new=0):
        assert token_is_expired()

    # expires_on is in history - means it is expired
    history_unix_time = time.time() - 100
    with patch(
        "ols.src.llms.providers.azure_openai.TokenCache.expires_on",
        new=history_unix_time,
    ):
        assert token_is_expired()

    # expires_on is in future - means it is not expired
    future_unix_time = time.time() + 100
    with patch(
        "ols.src.llms.providers.azure_openai.TokenCache.expires_on",
        new=future_unix_time,
    ):
        assert not token_is_expired()


def test_token_is_not_reused(provider_config, mocked_token_cache):
    """Test that token is not reused if it is expired."""
    retrieved_access_token = AccessToken(token="new_token", expires_on=0)  # noqa: S106

    with (
        patch(
            "ols.src.llms.providers.azure_openai.AzureOpenAI.retrieve_access_token",
            return_value=retrieved_access_token,
        ),
        patch(
            "ols.src.llms.providers.azure_openai.token_is_expired", return_value=True
        ),
        patch("ols.src.llms.providers.azure_openai.TokenCache", new=mocked_token_cache),
    ):
        access_token = AzureOpenAI(
            model="irrelevant value", provider_config=provider_config
        ).resolve_access_token("irrelevant value")
        assert access_token == "new_token"  # noqa: S105
        assert access_token == mocked_token_cache.access_token  # cache is updated


def test_token_is_reused(provider_config, mocked_token_cache):
    """Test that token is reused if it is not expired."""
    retrieved_access_token = AccessToken(token="new_token", expires_on=0)  # noqa: S106

    with (
        patch(
            "ols.src.llms.providers.azure_openai.AzureOpenAI.retrieve_access_token",
            return_value=retrieved_access_token,
        ),
        patch(
            "ols.src.llms.providers.azure_openai.token_is_expired", return_value=False
        ),
        patch("ols.src.llms.providers.azure_openai.TokenCache", new=mocked_token_cache),
    ):
        access_token = AzureOpenAI(
            model="irrelevant value", provider_config=provider_config
        ).resolve_access_token("irrelevant value")
        assert access_token == "cached_token"  # noqa: S105 old value
        assert access_token == mocked_token_cache.access_token  # cache is not updated
