"""Unit tests for Watsonx provider."""

from unittest.mock import MagicMock, patch

import pytest
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from ols.app.models.config import ProviderConfig
from ols.constants import GenericLLMParameters
from ols.src.llms.providers.watsonx import Watsonx, is_ibm_cloud_watsonx_url
from tests.mock_classes.mock_watsonxllm import ChatWatsonx


@pytest.fixture
def provider_config():
    """Fixture with provider configuration for Watsonx."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "watsonx",
            "url": "https://us-south.ml.cloud.ibm.com",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "01234567-89ab-cdef-0123-456789abcdef",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test_model_url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_credentials_directory():
    """Fixture with provider configuration for Watsonx."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "watsonx",
            "url": "https://us-south.ml.cloud.ibm.com",
            "credentials_path": "tests/config/secret",
            "project_id": "01234567-89ab-cdef-0123-456789abcdef",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test_model_url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_without_credentials():
    """Fixture with provider configuration for Watsonx without credentials."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "watsonx",
            "url": "https://us-south.ml.cloud.ibm.com",
            "project_id": "01234567-89ab-cdef-0123-456789abcdef",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test_model_url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_with_specific_params():
    """Fixture with provider configuration for Watsonx with provider-specific parameters."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "watsonx",
            "url": "https://us-south.ml.cloud.ibm.com",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "01234567-89ab-cdef-0123-456789abcdef",
            "watsonx_config": {
                "url": "https://eu-de.ml.cloud.ibm.com",
                "credentials_path": "tests/config/secret2/apitoken",
                "project_id": "ffffffff-89ab-cdef-0123-456789abcdef",
            },
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test_model_url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )


def test_basic_interface(provider_config):
    """Test basic interface."""
    with patch("ols.src.llms.providers.watsonx.ChatWatsonx", new=ChatWatsonx()):
        watsonx = Watsonx(model="uber-model", params={}, provider_config=provider_config)
        llm = watsonx.load()
        assert isinstance(llm, ChatWatsonx)
        assert watsonx.default_params


def test_params_handling(provider_config):
    """Test that not allowed parameters are removed before model init."""
    # first two parameters should be removed before model init
    # rest need to stay
    params = {
        "unknown_parameter": "foo",
        "verbose": True,
        "min_new_tokens": 1,
        "max_new_tokens": 10,
        "temperature": 0.3,
    }

    with patch("ols.src.llms.providers.watsonx.ChatWatsonx", new=ChatWatsonx()):
        watsonx = Watsonx(model="uber-model", params=params, provider_config=provider_config)
        llm = watsonx.load()
        assert isinstance(llm, ChatWatsonx)
        assert watsonx.default_params
        assert watsonx.params

        # taken from configuration
        assert watsonx.url == "https://us-south.ml.cloud.ibm.com"
        assert watsonx.credentials == "secret_key"
        assert watsonx.project_id == "01234567-89ab-cdef-0123-456789abcdef"

        # known parameters should be there
        assert GenParams.DECODING_METHOD in watsonx.params
        assert watsonx.params[GenParams.DECODING_METHOD] == "sample"

        assert GenParams.MAX_NEW_TOKENS in watsonx.params
        assert watsonx.params[GenParams.MAX_NEW_TOKENS] == 10

        # unknown parameters should be filtered out
        assert "unknown_parameter" not in watsonx.params
        assert "verbose" not in watsonx.params


def test_credentials_key_in_directory_handling(provider_config_credentials_directory):
    """Test that credentials in directory is handled as expected."""
    params = {}

    with patch("ols.src.llms.providers.watsonx.ChatWatsonx", new=ChatWatsonx()):
        watsonx = Watsonx(
            model="uber-model",
            params=params,
            provider_config=provider_config_credentials_directory,
        )
        llm = watsonx.load()
        assert isinstance(llm, ChatWatsonx)

        # taken from configuration
        assert watsonx.credentials == "secret_key"


def test_params_handling_specific_params(provider_config_with_specific_params):
    """Test that provider-specific parameters take precedence."""
    with patch("ols.src.llms.providers.watsonx.ChatWatsonx", new=ChatWatsonx()):
        watsonx = Watsonx(
            model="uber-model",
            params={},
            provider_config=provider_config_with_specific_params,
        )
        llm = watsonx.load()
        assert isinstance(llm, ChatWatsonx)
        assert watsonx.default_params
        assert watsonx.params

        # parameters taken from provier-specific configuration
        # which takes precedence over regular configuration
        assert watsonx.url == "https://eu-de.ml.cloud.ibm.com/"
        assert watsonx.credentials == "secret_key_2"
        assert watsonx.project_id == "ffffffff-89ab-cdef-0123-456789abcdef"


def test_params_handling_none_values(provider_config):
    """Test handling parameters with None values."""
    # first three parameters should be removed before model init
    # rest need to stay
    params = {
        "unknown_parameter": None,
        "temperature": None,
        "verbose": None,
        "min_new_tokens": None,
        "max_new_tokens": None,
    }

    with patch("ols.src.llms.providers.watsonx.ChatWatsonx", new=ChatWatsonx()):
        watsonx = Watsonx(model="uber-model", params=params, provider_config=provider_config)
        llm = watsonx.load()
        assert isinstance(llm, ChatWatsonx)
        assert watsonx.default_params
        assert watsonx.params

        # known parameters should be there
        assert GenParams.MIN_NEW_TOKENS in watsonx.params
        assert watsonx.params[GenParams.MIN_NEW_TOKENS] is None

        assert GenParams.MAX_NEW_TOKENS in watsonx.params
        assert watsonx.params[GenParams.MAX_NEW_TOKENS] is None

        assert GenParams.TEMPERATURE in watsonx.params
        assert watsonx.params[GenParams.TEMPERATURE] is None

        # unknown parameters should be filtered out
        assert "unknown_parameter" not in watsonx.params
        assert "verbose" not in watsonx.params


def test_params_replace_default_values_with_none(provider_config):
    """Test if default values are replaced by None values."""
    with patch("ols.src.llms.providers.watsonx.ChatWatsonx", new=ChatWatsonx()):
        # provider initialization with empty set of params
        watsonx = Watsonx(model="uber-model", params={}, provider_config=provider_config)
        watsonx.load()

        # check default value
        assert GenParams.DECODING_METHOD in watsonx.params
        assert watsonx.params[GenParams.DECODING_METHOD] == "sample"

        # provider initialization where default parameter is overriden
        params = {"decoding_method": None}

        watsonx = Watsonx(model="uber-model", params=params, provider_config=provider_config)
        watsonx.load()

        # check default value overrided by None
        assert GenParams.DECODING_METHOD in watsonx.params
        assert watsonx.params[GenParams.DECODING_METHOD] is None


def test_generic_parameter_mappings(provider_config):
    """Test generic parameter mapping to provider parameter list."""
    # some non-default values for generic LLM parameters
    generic_llm_params = {
        GenericLLMParameters.MIN_TOKENS_FOR_RESPONSE: 100,
        GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: 200,
        GenericLLMParameters.TOP_K: 10,
        GenericLLMParameters.TOP_P: 1.5,
        GenericLLMParameters.TEMPERATURE: 42.0,
    }

    with patch("ols.src.llms.providers.watsonx.ChatWatsonx", new=ChatWatsonx()):
        watsonx = Watsonx(
            model="uber-model",
            params=generic_llm_params,
            provider_config=provider_config,
        )
        llm = watsonx.load()
        assert isinstance(llm, ChatWatsonx)
        assert watsonx.default_params
        assert watsonx.params

        # generic parameters should be remapped to Watsonx-specific parameters
        assert GenParams.MIN_NEW_TOKENS in watsonx.params
        assert GenParams.MAX_NEW_TOKENS in watsonx.params
        assert GenParams.TOP_K in watsonx.params
        assert GenParams.TOP_P in watsonx.params
        assert GenParams.TEMPERATURE in watsonx.params
        assert watsonx.params[GenParams.MIN_NEW_TOKENS] == 100
        assert watsonx.params[GenParams.MAX_NEW_TOKENS] == 200
        assert watsonx.params[GenParams.TOP_K] == 10
        assert watsonx.params[GenParams.TOP_P] == 1.5
        assert watsonx.params[GenParams.TEMPERATURE] == 42.0


def test_missing_credentials_check(provider_config_without_credentials):
    """Test that check for missing credentials is in place ."""
    watsonx = Watsonx(
        model="uber-model",
        params={},
        provider_config=provider_config_without_credentials,
    )
    with pytest.raises(ValueError, match="Credentials must be specified"):
        watsonx.load()


def test_missing_project_id_check(provider_config):
    """Test that check for missing project ID is in place ."""
    watsonx = Watsonx(model="uber-model", params={}, provider_config=provider_config)
    # simulate situation when project ID is missing
    watsonx.provider_config.project_id = None
    with pytest.raises(ValueError, match="Project ID must be specified"):
        watsonx.load()


def _cpd_secret_dir(tmp_path, *, username=True, version=True, instance_id=False):
    secret_dir = tmp_path / "watsonx_cpd"
    secret_dir.mkdir()
    (secret_dir / "apitoken").write_text("cpd_api_key")
    if username:
        (secret_dir / "username").write_text("cpd_user")
    if version:
        (secret_dir / "version").write_text("5.1")
    if instance_id:
        (secret_dir / "instance_id").write_text("openshift")
    return secret_dir


def _cpd_provider_config(secret_dir):
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "watsonx",
            "url": "https://cpd-instance.apps.example.com",
            "credentials_path": str(secret_dir),
            "project_id": "01234567-89ab-cdef-0123-456789abcdef",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test_model_url/",
                    "credentials_path": str(secret_dir / "apitoken"),
                }
            ],
        }
    )


def test_ibm_cloud_does_not_pass_cpd_fields(provider_config):
    """IBM Cloud watsonx still constructs ChatWatsonx with apikey only."""
    mock_cls = MagicMock()
    with patch("ols.src.llms.providers.watsonx.ChatWatsonx", mock_cls):
        Watsonx(model="uber-model", params={}, provider_config=provider_config).load()
    kwargs = mock_cls.call_args.kwargs
    assert kwargs["apikey"] == "secret_key"
    assert "username" not in kwargs
    assert "version" not in kwargs
    assert "instance_id" not in kwargs


def test_cpd_passes_username_version_and_instance_id(tmp_path):
    """CP4D URL must pass username, version, and instance_id into ChatWatsonx."""
    provider_config = _cpd_provider_config(_cpd_secret_dir(tmp_path, instance_id=True))
    mock_cls = MagicMock()
    with patch("ols.src.llms.providers.watsonx.ChatWatsonx", mock_cls):
        Watsonx(model="uber-model", params={}, provider_config=provider_config).load()
    kwargs = mock_cls.call_args.kwargs
    assert kwargs["apikey"] == "cpd_api_key"
    assert kwargs["username"] == "cpd_user"
    assert kwargs["version"] == "5.1"
    assert kwargs["instance_id"] == "openshift"


def test_cpd_defaults_instance_id_when_missing(tmp_path):
    """CP4D without instance_id still works; default is openshift."""
    provider_config = _cpd_provider_config(_cpd_secret_dir(tmp_path))
    mock_cls = MagicMock()
    with patch("ols.src.llms.providers.watsonx.ChatWatsonx", mock_cls):
        Watsonx(model="uber-model", params={}, provider_config=provider_config).load()
    assert mock_cls.call_args.kwargs["instance_id"] == "openshift"


def test_cpd_missing_username(tmp_path):
    """CP4D URL without username must fail clearly, not with WATSONX_USERNAME."""
    provider_config = _cpd_provider_config(_cpd_secret_dir(tmp_path, username=False))
    watsonx = Watsonx(model="uber-model", params={}, provider_config=provider_config)
    with pytest.raises(ValueError, match="username"):
        watsonx.load()


def test_cpd_missing_version(tmp_path):
    """CP4D URL without version must fail clearly (OLS-2849)."""
    provider_config = _cpd_provider_config(_cpd_secret_dir(tmp_path, version=False))
    watsonx = Watsonx(model="uber-model", params={}, provider_config=provider_config)
    with pytest.raises(ValueError, match="version"):
        watsonx.load()


def test_is_ibm_cloud_watsonx_url():
    """IBM Cloud SaaS hosts keep the apitoken-only path."""
    assert is_ibm_cloud_watsonx_url("https://us-south.ml.cloud.ibm.com")
    assert is_ibm_cloud_watsonx_url("https://eu-de.ml.cloud.ibm.com/ml")
    assert is_ibm_cloud_watsonx_url("")
    assert not is_ibm_cloud_watsonx_url("https://cpd-instance.apps.example.com")
    assert not is_ibm_cloud_watsonx_url("https://watsonx.example.com")
