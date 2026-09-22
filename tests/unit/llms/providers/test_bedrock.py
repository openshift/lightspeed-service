"""Unit tests for AWS Bedrock provider."""

from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest

from ols import config
from ols.app.models.config import ProviderConfig
from ols.src.llms.llm_loader import LLMConfigurationError
from ols.src.llms.providers.bedrock import Bedrock


@pytest.fixture(autouse=True)
def _isolate_bedrock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear Bedrock-related env vars before each test."""
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)


@pytest.fixture
def aws_creds_dir(tmp_path: Path) -> str:
    """Create a temporary directory with fake AWS IAM credential files."""
    creds_dir = tmp_path / "aws_creds"
    creds_dir.mkdir()
    (creds_dir / "aws_access_key_id").write_text("test_access_key")
    (creds_dir / "aws_secret_access_key").write_text("test_secret_key")
    (creds_dir / "role_arn").write_text("arn:aws:iam::123456789012:role/TestRole")
    return str(creds_dir)


@pytest.fixture
def provider_config() -> ProviderConfig:
    """Fixture with provider configuration for Bedrock (Bearer token)."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "bedrock",
            "url": "https://bedrock-mantle.us-east-1.api.aws",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "anthropic.claude-opus-4-7",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_iam(aws_creds_dir: str) -> ProviderConfig:
    """Fixture with provider configuration for Bedrock (IAM credentials)."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "bedrock",
            "url": "https://bedrock-mantle.us-east-1.api.aws",
            "credentials_path": aws_creds_dir,
            "models": [
                {
                    "name": "anthropic.claude-opus-4-7",
                }
            ],
        }
    )


# --- Bearer token tests ---


@patch(
    "ols.src.llms.providers.bedrock.ChatBedrockConverse",
    autospec=True,
)
def test_load_anthropic_model(
    mock_chat: MagicMock, provider_config: ProviderConfig
) -> None:
    """Test that anthropic-prefixed models use ChatBedrockConverse."""
    bedrock = Bedrock(
        model="anthropic.claude-opus-4-7", params={}, provider_config=provider_config
    )
    llm = bedrock.load()
    assert llm is not None

    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["model_id"] == "us.anthropic.claude-opus-4-7"
    assert call_kwargs["region_name"] == "us-east-1"
    assert call_kwargs["max_tokens"] == 4096
    assert call_kwargs["bedrock_api_key"] == "secret_key"


@patch(
    "ols.src.llms.providers.provider.LLMProvider._construct_httpx_client",
    return_value=MagicMock(),
)
@patch(
    "ols.src.llms.providers.bedrock.ChatOpenAI",
    autospec=True,
)
def test_load_openai_model(
    mock_chat: MagicMock, _mock_httpx: MagicMock, provider_config: ProviderConfig
) -> None:
    """Test that openai-prefixed models use ChatOpenAI with Responses API."""
    bedrock = Bedrock(
        model="openai.gpt-5.4", params={}, provider_config=provider_config
    )
    llm = bedrock.load()
    assert llm is not None

    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["model"] == "openai.gpt-5.4"
    assert (
        call_kwargs["base_url"] == "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
    )
    assert call_kwargs["openai_api_key"] == "secret_key"
    assert call_kwargs["use_responses_api"] is True
    assert call_kwargs["max_tokens"] == 4096


@patch(
    "ols.src.llms.providers.provider.LLMProvider._construct_httpx_client",
    return_value=MagicMock(),
)
@patch(
    "ols.src.llms.providers.bedrock.ChatOpenAI",
    autospec=True,
)
def test_load_default_model(
    mock_chat: MagicMock, _mock_httpx: MagicMock, provider_config: ProviderConfig
) -> None:
    """Test that non-prefixed models use ChatOpenAI with Chat Completions API."""
    bedrock = Bedrock(model="deepseek.v3.1", params={}, provider_config=provider_config)
    llm = bedrock.load()
    assert llm is not None

    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["model"] == "deepseek.v3.1"
    assert call_kwargs["base_url"] == "https://bedrock-mantle.us-east-1.api.aws/v1"
    assert call_kwargs["openai_api_key"] == "secret_key"
    assert call_kwargs["use_responses_api"] is False
    assert call_kwargs["max_tokens"] == 4096


def test_default_params(provider_config: ProviderConfig) -> None:
    """Test default parameters are populated correctly."""
    bedrock = Bedrock(
        model="anthropic.claude-opus-4-7", params={}, provider_config=provider_config
    )
    defaults = bedrock.default_params
    assert defaults["api_key"] == "secret_key"
    assert defaults["model"] == "anthropic.claude-opus-4-7"
    assert "temperature" in defaults
    assert "max_tokens" in defaults


def test_default_params_missing_url() -> None:
    """Test that missing URL raises LLMConfigurationError."""
    provider_config = ProviderConfig(
        {
            "name": "some_provider",
            "type": "bedrock",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "anthropic.claude-opus-4-7",
                }
            ],
        }
    )
    with pytest.raises(LLMConfigurationError, match="url is required"):
        Bedrock(
            model="anthropic.claude-opus-4-7",
            params={},
            provider_config=provider_config,
        )


def test_default_params_missing_credentials() -> None:
    """Test that missing credentials raises LLMConfigurationError."""
    provider_config = ProviderConfig(
        {
            "name": "some_provider",
            "type": "bedrock",
            "url": "https://bedrock-mantle.us-east-1.api.aws",
            "models": [
                {
                    "name": "anthropic.claude-opus-4-7",
                }
            ],
        }
    )
    with pytest.raises(LLMConfigurationError, match="credentials are required"):
        Bedrock(
            model="anthropic.claude-opus-4-7",
            params={},
            provider_config=provider_config,
        )


@patch(
    "ols.src.llms.providers.bedrock.ChatBedrockConverse",
    autospec=True,
)
def test_params_handling(mock_chat: MagicMock, provider_config: ProviderConfig) -> None:
    """Test that disallowed parameters are filtered before model init."""
    params = {
        "unknown_parameter": "foo",
        "min_new_tokens": 1,
        "max_new_tokens": 10,
        "temperature": 0.3,
    }

    bedrock = Bedrock(
        model="anthropic.claude-opus-4-7",
        params=params,
        provider_config=provider_config,
    )
    llm = bedrock.load()
    assert llm is not None
    assert bedrock.params

    assert "temperature" in bedrock.params
    assert bedrock.params["temperature"] == 0.3

    assert "min_new_tokens" not in bedrock.params
    assert "max_new_tokens" not in bedrock.params
    assert "unknown_parameter" not in bedrock.params


@patch(
    "ols.src.llms.providers.provider.LLMProvider._construct_httpx_client",
    return_value=MagicMock(),
)
@patch(
    "ols.src.llms.providers.bedrock.ChatOpenAI",
    autospec=True,
)
def test_max_tokens_remapped_for_openai_models(
    mock_chat: MagicMock, _mock_httpx: MagicMock, provider_config: ProviderConfig
) -> None:
    """Test that max_tokens is passed through for ChatOpenAI."""
    bedrock = Bedrock(
        model="openai.gpt-5.4",
        params={"max_tokens": 1024},
        provider_config=provider_config,
    )
    bedrock.load()

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["max_tokens"] == 1024


@patch(
    "ols.src.llms.providers.provider.LLMProvider._construct_httpx_client",
    return_value=MagicMock(),
)
@patch(
    "ols.src.llms.providers.bedrock.ChatOpenAI",
    autospec=True,
)
def test_max_tokens_remapped_for_default_models(
    mock_chat: MagicMock, _mock_httpx: MagicMock, provider_config: ProviderConfig
) -> None:
    """Test that max_tokens is passed through for default route."""
    bedrock = Bedrock(
        model="deepseek.v3.1",
        params={"max_tokens": 2048},
        provider_config=provider_config,
    )
    bedrock.load()

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["max_tokens"] == 2048


@patch(
    "ols.src.llms.providers.bedrock.ChatBedrockConverse",
    autospec=True,
)
def test_anthropic_passes_temperature(
    mock_chat: MagicMock, provider_config: ProviderConfig
) -> None:
    """Test that temperature is passed through to ChatBedrockConverse."""
    bedrock = Bedrock(
        model="anthropic.claude-opus-4-7",
        params={"temperature": 0.5},
        provider_config=provider_config,
    )
    bedrock.load()

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["temperature"] == 0.5


@patch(
    "ols.src.llms.providers.bedrock.ChatBedrockConverse",
    autospec=True,
)
def test_temperature_stripped_when_not_supported(
    mock_chat: MagicMock,
) -> None:
    """Test that temperature is omitted when model sets temperature_supported=False."""
    pc = ProviderConfig(
        {
            "name": "some_provider",
            "type": "bedrock",
            "url": "https://bedrock-mantle.us-east-1.api.aws",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "anthropic.claude-sonnet-5",
                    "parameters": {"temperature_supported": False},
                }
            ],
        }
    )
    bedrock = Bedrock(
        model="anthropic.claude-sonnet-5",
        params={},
        provider_config=pc,
    )
    assert "temperature" not in bedrock.params

    bedrock.load()
    call_kwargs = mock_chat.call_args[1]
    assert "temperature" not in call_kwargs


@patch(
    "ols.src.llms.providers.bedrock.ChatBedrockConverse",
    autospec=True,
)
def test_temperature_present_when_supported(
    mock_chat: MagicMock, provider_config: ProviderConfig
) -> None:
    """Test that temperature is present by default (temperature_supported=True)."""
    bedrock = Bedrock(
        model="anthropic.claude-opus-4-7",
        params={},
        provider_config=provider_config,
    )
    assert "temperature" in bedrock.params

    bedrock.load()
    call_kwargs = mock_chat.call_args[1]
    assert "temperature" in call_kwargs


@patch(
    "ols.src.llms.providers.bedrock.ChatBedrockConverse",
    autospec=True,
)
def test_temperature_stripped_even_when_caller_passes_it(
    mock_chat: MagicMock,
) -> None:
    """Test that caller-supplied temperature is also stripped when not supported."""
    pc = ProviderConfig(
        {
            "name": "some_provider",
            "type": "bedrock",
            "url": "https://bedrock-mantle.us-east-1.api.aws",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "anthropic.claude-sonnet-5",
                    "parameters": {"temperature_supported": False},
                }
            ],
        }
    )
    bedrock = Bedrock(
        model="anthropic.claude-sonnet-5",
        params={"temperature": 0.5},
        provider_config=pc,
    )
    assert "temperature" not in bedrock.params

    bedrock.load()
    call_kwargs = mock_chat.call_args[1]
    assert "temperature" not in call_kwargs


@patch(
    "ols.src.llms.providers.bedrock.ChatBedrockConverse",
    autospec=True,
)
def test_temperature_stripped_when_set_via_dev_config(
    mock_chat: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that dev_config temperature override is stripped when model does not support it."""
    monkeypatch.setattr(config.dev_config, "llm_params", {"temperature": 0.7})
    pc = ProviderConfig(
        {
            "name": "some_provider",
            "type": "bedrock",
            "url": "https://bedrock-mantle.us-east-1.api.aws",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "anthropic.claude-sonnet-5",
                    "parameters": {"temperature_supported": False},
                }
            ],
        }
    )
    bedrock = Bedrock(
        model="anthropic.claude-sonnet-5",
        params={},
        provider_config=pc,
    )
    assert "temperature" not in bedrock.params

    bedrock.load()
    call_kwargs = mock_chat.call_args[1]
    assert "temperature" not in call_kwargs


def test_region_extraction(provider_config: ProviderConfig) -> None:
    """Test that region is correctly extracted from the Mantle URL."""
    bedrock = Bedrock(
        model="anthropic.claude-opus-4-7", params={}, provider_config=provider_config
    )
    assert bedrock._region_from_url() == "us-east-1"


def test_region_extraction_invalid_url() -> None:
    """Test that invalid URL raises LLMConfigurationError during region extraction."""
    provider_config = ProviderConfig(
        {
            "name": "some_provider",
            "type": "bedrock",
            "url": "https://invalid-url",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [{"name": "anthropic.claude-opus-4-7"}],
        }
    )
    bedrock = Bedrock(
        model="anthropic.claude-opus-4-7", params={}, provider_config=provider_config
    )
    with pytest.raises(LLMConfigurationError, match="cannot extract region"):
        bedrock._region_from_url()


# --- IAM credential tests ---


@patch(
    "ols.src.llms.providers.bedrock.Bedrock._build_boto3_session",
)
@patch(
    "ols.src.llms.providers.bedrock.ChatBedrockConverse",
    autospec=True,
)
def test_load_anthropic_model_iam(
    mock_chat: MagicMock,
    mock_build_session: MagicMock,
    provider_config_iam: ProviderConfig,
) -> None:
    """Test Anthropic route with IAM credentials passes a boto3 client."""
    mock_client = MagicMock()
    mock_build_session.return_value.client.return_value = mock_client

    bedrock = Bedrock(
        model="anthropic.claude-opus-4-7",
        params={},
        provider_config=provider_config_iam,
    )
    bedrock.load()

    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["model_id"] == "us.anthropic.claude-opus-4-7"
    assert call_kwargs["region_name"] == "us-east-1"
    assert call_kwargs["client"] is mock_client
    assert "bedrock_api_key" not in call_kwargs
    mock_build_session.assert_called_once_with("us-east-1", ANY)
    mock_build_session.return_value.client.assert_called_once_with("bedrock-runtime")


@patch(
    "ols.src.llms.providers.bedrock.ChatOpenAI",
    autospec=True,
)
@patch(
    "ols.src.llms.providers.bedrock.Bedrock._build_sigv4_auth",
    return_value=MagicMock(),
)
def test_load_openai_model_iam(
    _mock_sigv4: MagicMock,
    mock_chat: MagicMock,
    provider_config_iam: ProviderConfig,
) -> None:
    """Test OpenAI route with IAM credentials uses SigV4 auth."""
    bedrock = Bedrock(
        model="openai.gpt-5.4", params={}, provider_config=provider_config_iam
    )
    bedrock.load()

    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["openai_api_key"] == "unused"
    assert call_kwargs["use_responses_api"] is True
    assert "http_client" in call_kwargs
    assert "http_async_client" in call_kwargs
    _mock_sigv4.assert_called_once_with("us-east-1", ANY)


@patch(
    "ols.src.llms.providers.bedrock.ChatOpenAI",
    autospec=True,
)
@patch(
    "ols.src.llms.providers.bedrock.Bedrock._build_sigv4_auth",
    return_value=MagicMock(),
)
def test_load_default_model_iam(
    _mock_sigv4: MagicMock,
    mock_chat: MagicMock,
    provider_config_iam: ProviderConfig,
) -> None:
    """Test default route with IAM credentials uses SigV4 auth."""
    bedrock = Bedrock(
        model="deepseek.v3.1", params={}, provider_config=provider_config_iam
    )
    bedrock.load()

    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["openai_api_key"] == "unused"
    assert call_kwargs["use_responses_api"] is False


def test_has_aws_credentials(provider_config_iam: ProviderConfig) -> None:
    """Test _has_aws_credentials returns True when IAM creds are present."""
    bedrock = Bedrock(
        model="anthropic.claude-opus-4-7",
        params={},
        provider_config=provider_config_iam,
    )
    assert bedrock._has_aws_credentials() is True


def test_has_aws_credentials_false(provider_config: ProviderConfig) -> None:
    """Test _has_aws_credentials returns False when only Bearer token is set."""
    bedrock = Bedrock(
        model="anthropic.claude-opus-4-7", params={}, provider_config=provider_config
    )
    assert bedrock._has_aws_credentials() is False


@patch("ols.src.llms.providers.bedrock.boto3")
def test_build_sigv4_auth_without_role(
    mock_boto3: MagicMock, provider_config_iam: ProviderConfig
) -> None:
    """Test _build_sigv4_auth without role_arn uses direct credentials."""
    provider_config_iam.role_arn = None

    mock_frozen = MagicMock()
    mock_frozen.access_key = "AKID"
    mock_frozen.secret_key = "SECRET"  # noqa: S105
    mock_frozen.token = None
    mock_session = MagicMock()
    mock_session.get_credentials.return_value.get_frozen_credentials.return_value = (
        mock_frozen
    )
    mock_boto3.Session.return_value = mock_session

    bedrock = Bedrock(
        model="openai.gpt-5.4", params={}, provider_config=provider_config_iam
    )
    auth = bedrock._build_sigv4_auth("us-east-1")
    assert auth is not None
    mock_boto3.Session.assert_called_once_with(
        aws_access_key_id="test_access_key",
        aws_secret_access_key="test_secret_key",  # noqa: S106
        region_name="us-east-1",
    )


@patch("ols.src.llms.providers.bedrock.boto3")
def test_build_sigv4_auth_with_role(
    mock_boto3: MagicMock, provider_config_iam: ProviderConfig
) -> None:
    """Test _build_sigv4_auth with role_arn uses STS assume_role."""
    mock_sts = MagicMock()
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "ASSUMED_AKID",
            "SecretAccessKey": "ASSUMED_SECRET",
            "SessionToken": "ASSUMED_TOKEN",
        }
    }
    mock_frozen = MagicMock()
    mock_frozen.access_key = "ASSUMED_AKID"
    mock_frozen.secret_key = "ASSUMED_SECRET"  # noqa: S105
    mock_frozen.token = "ASSUMED_TOKEN"  # noqa: S105
    mock_session = MagicMock()
    mock_session.client.return_value = mock_sts
    mock_session.get_credentials.return_value.get_frozen_credentials.return_value = (
        mock_frozen
    )
    mock_boto3.Session.return_value = mock_session

    bedrock = Bedrock(
        model="openai.gpt-5.4", params={}, provider_config=provider_config_iam
    )
    auth = bedrock._build_sigv4_auth("us-east-1")
    assert auth is not None
    mock_sts.assume_role.assert_called_once_with(
        RoleArn="arn:aws:iam::123456789012:role/TestRole",
        RoleSessionName="ols-bedrock",
    )


def test_anthropic_thinking_with_thinking_effort() -> None:
    """Test Bedrock Anthropic reasoning with thinking_effort parameter."""
    provider_config = ProviderConfig(
        {
            "name": "some_provider",
            "type": "bedrock",
            "url": "https://bedrock-mantle.us-east-1.api.aws",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "anthropic.claude-opus-4-7",
                    "parameters": {
                        "reasoning_config": {
                            "thinking_effort": "high",
                        }
                    },
                }
            ],
        }
    )
    bedrock = Bedrock(
        model="anthropic.claude-opus-4-7", params={}, provider_config=provider_config
    )

    params = {**bedrock.params}
    model_params = bedrock.provider_config.models.get(bedrock.model)
    from ols.app.models.config import ModelParameters

    model_params_obj = getattr(model_params, "parameters", None) or ModelParameters()
    bedrock._configure_anthropic_thinking(params, model_params_obj, 4096)

    assert "additional_model_request_fields" in params
    assert "thinking" in params["additional_model_request_fields"]
    assert (
        params["additional_model_request_fields"]["thinking"]["reasoning_effort"]
        == "high"
    )
    # Sampling parameters should be removed when thinking is enabled
    assert "temperature" not in params
    assert "top_p" not in params
    assert "top_k" not in params


def test_anthropic_thinking_with_type_and_budget_tokens() -> None:
    """Test Bedrock Anthropic reasoning fallback with type and budget_tokens."""
    provider_config = ProviderConfig(
        {
            "name": "some_provider",
            "type": "bedrock",
            "url": "https://bedrock-mantle.us-east-1.api.aws",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "anthropic.claude-opus-4-7",
                    "parameters": {
                        "reasoning_config": {
                            "type": "enabled",
                            "budget_tokens": 5000,
                        }
                    },
                }
            ],
        }
    )
    bedrock = Bedrock(
        model="anthropic.claude-opus-4-7", params={}, provider_config=provider_config
    )

    params = {**bedrock.params}
    model_params = bedrock.provider_config.models.get(bedrock.model)
    from ols.app.models.config import ModelParameters

    model_params_obj = getattr(model_params, "parameters", None) or ModelParameters()
    bedrock._configure_anthropic_thinking(params, model_params_obj, 4096)

    assert "additional_model_request_fields" in params
    assert "thinking" in params["additional_model_request_fields"]
    assert params["additional_model_request_fields"]["thinking"]["type"] == "enabled"
    # Budget tokens should be clamped to [1024, 25% of max_tokens]
    assert (
        params["additional_model_request_fields"]["thinking"]["budget_tokens"] == 1024
    )
    # Sampling parameters should be removed
    assert "temperature" not in params
    assert "top_p" not in params
    assert "top_k" not in params


def test_anthropic_thinking_budget_tokens_clamping() -> None:
    """Test Bedrock Anthropic budget_tokens is clamped to valid range."""
    provider_config = ProviderConfig(
        {
            "name": "some_provider",
            "type": "bedrock",
            "url": "https://bedrock-mantle.us-east-1.api.aws",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "anthropic.claude-opus-4-7",
                    "parameters": {
                        "reasoning_config": {
                            "type": "enabled",
                            "budget_tokens": 50000,  # Exceeds max_tokens * 0.25
                        }
                    },
                }
            ],
        }
    )
    bedrock = Bedrock(
        model="anthropic.claude-opus-4-7", params={}, provider_config=provider_config
    )

    params = {**bedrock.params}
    model_params = bedrock.provider_config.models.get(bedrock.model)
    from ols.app.models.config import ModelParameters

    model_params_obj = getattr(model_params, "parameters", None) or ModelParameters()
    max_tokens = 4096
    bedrock._configure_anthropic_thinking(params, model_params_obj, max_tokens)

    # Budget should be clamped to 25% of 4096 = 1024
    expected_max = int(max_tokens * 0.25)
    assert (
        params["additional_model_request_fields"]["thinking"]["budget_tokens"]
        == expected_max
    )


def test_anthropic_no_thinking_config() -> None:
    """Test Bedrock Anthropic with no reasoning_config."""
    provider_config = ProviderConfig(
        {
            "name": "some_provider",
            "type": "bedrock",
            "url": "https://bedrock-mantle.us-east-1.api.aws",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "anthropic.claude-opus-4-7",
                }
            ],
        }
    )
    bedrock = Bedrock(
        model="anthropic.claude-opus-4-7", params={}, provider_config=provider_config
    )

    params = {**bedrock.params}
    from ols.app.models.config import ModelParameters

    model_params_obj = ModelParameters()
    bedrock._configure_anthropic_thinking(params, model_params_obj, 4096)

    # Should return early without setting anything
    assert "additional_model_request_fields" not in params


def test_bedrock_picks_up_rotated_credentials(tmp_path: Path) -> None:
    """Test that Bedrock provider re-reads credentials when hot-reload is enabled."""
    secret_file = tmp_path / "apitoken"
    secret_file.write_text("initial-key")

    config = ProviderConfig(
        {
            "name": "test_provider",
            "type": "bedrock",
            "url": "https://bedrock-mantle.us-east-1.api.aws",
            "credentials_path": str(secret_file),
            "models": [{"name": "anthropic.claude-opus-4-7"}],
        },
        credential_hot_reload=True,
    )

    bedrock_1 = Bedrock(
        model="anthropic.claude-opus-4-7", params={}, provider_config=config
    )
    assert bedrock_1.default_params["api_key"] == "initial-key"

    secret_file.write_text("rotated-key")

    bedrock_2 = Bedrock(
        model="anthropic.claude-opus-4-7", params={}, provider_config=config
    )
    assert bedrock_2.default_params["api_key"] == "rotated-key"


def test_bedrock_iam_picks_up_rotated_credentials(tmp_path: Path) -> None:
    """Test that Bedrock IAM credentials are re-read when hot-reload is enabled."""
    creds_dir = tmp_path / "aws_creds"
    creds_dir.mkdir()
    (creds_dir / "aws_access_key_id").write_text("original_access")
    (creds_dir / "aws_secret_access_key").write_text("original_secret")

    config = ProviderConfig(
        {
            "name": "test_provider",
            "type": "bedrock",
            "url": "https://bedrock-mantle.us-east-1.api.aws",
            "credentials_path": str(creds_dir),
            "models": [{"name": "anthropic.claude-opus-4-7"}],
        },
        credential_hot_reload=True,
    )

    access_key, secret_key, _ = config.get_aws_credentials()
    assert access_key == "original_access"
    assert secret_key == "original_secret"  # noqa: S105

    (creds_dir / "aws_access_key_id").write_text("rotated_access")
    (creds_dir / "aws_secret_access_key").write_text("rotated_secret")

    access_key, secret_key, _ = config.get_aws_credentials()
    assert access_key == "rotated_access"
    assert secret_key == "rotated_secret"  # noqa: S105


def test_bedrock_api_key_not_misclassified_as_iam(tmp_path: Path) -> None:
    """Regression: API-key file must not be read as IAM credentials.

    When credentials_path is a file (API-key layout), get_aws_credentials()
    must not treat the API token as aws_access_key_id / aws_secret_access_key.
    """
    secret_file = tmp_path / "apitoken"
    secret_file.write_text("my-bedrock-api-key")

    config = ProviderConfig(
        {
            "name": "test_provider",
            "type": "bedrock",
            "url": "https://bedrock-mantle.us-east-1.api.aws",
            "credentials_path": str(secret_file),
            "models": [{"name": "anthropic.claude-opus-4-7"}],
        },
        credential_hot_reload=True,
    )

    access_key, secret_key, role_arn = config.get_aws_credentials()
    assert access_key is None
    assert secret_key is None
    assert role_arn is None


@patch("ols.src.llms.providers.bedrock.ChatBedrockConverse")
def test_bedrock_load_uses_api_key_path_with_file_credentials(
    mock_chat: MagicMock,
    tmp_path: Path,
) -> None:
    """Regression: Bedrock.load() must use API-key auth when credentials_path is a file."""
    secret_file = tmp_path / "apitoken"
    secret_file.write_text("my-bedrock-api-key")

    provider_config = ProviderConfig(
        {
            "name": "test_provider",
            "type": "bedrock",
            "url": "https://bedrock-mantle.us-east-1.api.aws",
            "credentials_path": str(secret_file),
            "models": [{"name": "anthropic.claude-opus-4-7"}],
        },
        credential_hot_reload=True,
    )

    bedrock = Bedrock(
        model="anthropic.claude-opus-4-7", params={}, provider_config=provider_config
    )
    bedrock.load()

    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["bedrock_api_key"] == "my-bedrock-api-key"
    assert "client" not in call_kwargs
