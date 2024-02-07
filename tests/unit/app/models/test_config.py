"""Unit tests for data models."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import AnyHttpUrl, ValidationError

from ols import constants
from ols.app.models.config import (
    Config,
    ConversationCacheConfig,
    DevConfig,
    LLMProviderConfig,
    LoggingConfig,
    MemoryConfig,
    ModelConfig,
    OLSConfig,
    PostgresConfig,
    RedisConfig,
    RedisCredentials,
    ReferenceContent,
    _get_attribute_from_file,
)


def test_model_config_instantiation():
    "Ensure ModelConfig instantiates correctly with a name." ""
    config = ModelConfig(name="Example Name")
    assert config.name == "Example Name"


def test_model_config_requires_name():
    """Verify ModelConfig raises ValidationError if name is missing."""
    try:
        ModelConfig()
    except ValidationError:
        assert True
    else:
        assert False


def test_model_config_field_type_validation():
    """Check ModelConfig name field for type validation."""
    try:
        ModelConfig(name=123)
    except ValidationError:
        assert True
    else:
        assert False


@pytest.fixture
def credentials_file(tmp_path):
    """Create a temporary credentials file."""
    file = tmp_path / "credentials.txt"
    file.write_text("user\npassword")
    return file


def test_get_attribute_from_file_success(credentials_file):
    """Test successful attribute retrieval from file."""
    assert _get_attribute_from_file(credentials_file) == "user\npassword"


def test_get_attribute_from_file_not_found():
    """Check handling of non-existent file."""
    with pytest.raises(ValueError) as excinfo:
        _get_attribute_from_file("non_existent_file.txt")
    assert "File not found at" in str(excinfo.value)


def test_llm_provider_config_init(credentials_file):
    """Verify LLMProviderConfig initialization with valid credentials."""
    config = LLMProviderConfig(
        url="https://example.com",
        credentials_path=credentials_file,
        models=[],
        type="bam",
    )
    assert config.credentials == "user\npassword"
    assert config.url == AnyHttpUrl("https://example.com")


def test_llm_provider_config_init_watsonx(credentials_file):
    """Verify LLMProviderConfig initialization with valid credentials."""
    config = LLMProviderConfig(
        url="https://example.com",
        credentials_path=credentials_file,
        models=[],
        type="watsonx",
        project_id="whatever?",
    )
    assert config.credentials == "user\npassword"
    assert config.url == AnyHttpUrl("https://example.com")
    assert config.type == "watsonx"
    assert config.project_id == "whatever?"


def test_llm_provider_config_init_watsonx_without_project_id(credentials_file):
    """Verify LLMProviderConfig initialization with valid credentials."""
    with pytest.raises(ValidationError) as excinfo:
        LLMProviderConfig(
            url="https://example.com",
            credentials_path=credentials_file,
            models=[],
            type="watsonx",
        )
    assert "project_id is required for watsonx provider" in str(excinfo.value)


def test_llm_provider_config_missing_credentials():
    """Test error on missing credentials_path."""
    with pytest.raises(ValidationError) as excinfo:
        LLMProviderConfig(url="http://example.com", models=[], type="bam")
    assert "credentials_path\n  Field required " in str(excinfo.value)


def test_llm_provider_config_invalid_credentials_path():
    """Checks error for invalid credentials path."""
    with pytest.raises(ValueError) as excinfo:
        LLMProviderConfig(
            url="http://example.com",
            credentials_path="non_existent_file.txt",
            models=[],
            type="bam",
        )
    assert "Path does not point to a file" in str(excinfo.value)


def test_provider_config_explicit_context_window_size():
    """Test the ProviderConfig model when explicit context window size is specified."""
    context_window_size = 500
    response_token_limit = 100

    provider_config = LLMProviderConfig(
        **{
            "name": "test_name",
            "type": "bam",
            "url": "http://example.com",
            "credentials_path": "tests/config/secret.txt",
            "project_id": "test_project_id",
            "models": [
                {
                    "name": "test_model_name",
                    "context_window_size": context_window_size,
                    "response_token_limit": response_token_limit,
                }
            ],
        }
    )
    assert provider_config.models[0].context_window_size == context_window_size


def test_provider_config_improper_context_window_size_value():
    """Test the ProviderConfig model when improper context window size is specified."""
    with pytest.raises(ValueError) as excinfo:
        LLMProviderConfig(
            **{
                "name": "test_name",
                "type": "bam",
                "url": "http://example.com",
                "credentials_path": "tests/config/secret.txt",
                "project_id": "test_project_id",
                "models": [
                    {
                        "name": "test_model_name",
                        "context_window_size": -1,
                    }
                ],
            }
        )
    assert "Input should be greater than 0" in str(excinfo.value)


def test_provider_config_improper_context_window_size_type():
    """Test the ProviderConfig model when improper context window size is specified."""
    with pytest.raises(ValueError) as excinfo:
        LLMProviderConfig(
            **{
                "name": "test_name",
                "type": "bam",
                "url": "http://example.com",
                "credentials_path": "tests/config/secret.txt",
                "project_id": "test_project_id",
                "models": [
                    {
                        "name": "test_model_name",
                        "context_window_size": "not_number",
                    }
                ],
            }
        )
    assert "context_window_size\n  Input should be a valid integer" in str(
        excinfo.value
    )


def test_llm_provider_config_invalid_url(credentials_file):
    """Verify URL validation in LLMProviderConfig."""
    with pytest.raises(ValidationError) as excinfo:
        LLMProviderConfig(
            url="invalid-url", credentials_path=credentials_file, models=[], type="bam"
        )
    assert "Input should be a valid URL" in str(excinfo.value)


def test_llm_provider_config_invalid_url_scheme(credentials_file):
    """Test URL scheme validation in LLMProviderConfig."""
    with pytest.raises(ValidationError) as excinfo:
        LLMProviderConfig(
            url="ftp://example.com",
            credentials_path=credentials_file,
            models=[],
            type="bam",
        )
    assert "URL scheme should be 'http' or 'https'" in str(excinfo.value)


@pytest.fixture
def user_file(tmp_path):
    """Create a temporary user file."""
    file = tmp_path / "user.txt"
    file.write_text("test_user")
    return file


@pytest.fixture
def password_file(tmp_path):
    """Create a temporary password file."""
    file = tmp_path / "password.txt"
    file.write_text("test_password")
    return file


def test_redis_credentials_init_success(user_file, password_file):
    """Verify successful RedisCredentials initialization."""
    credentials = RedisCredentials(user_path=user_file, password_path=password_file)
    assert credentials.username == "test_user"
    assert credentials.password == "test_password"  # noqa


def test_redis_credentials_user_path_not_found():
    """Check handling of non-existent user file path."""
    with pytest.raises(ValueError) as excinfo:
        RedisCredentials(
            user_path="non_existent_user_file.txt",
            password_path="dummy_path",  # noqa
        )
    assert "Path does not point to a file" in str(excinfo.value)


def test_redis_credentials_password_path_not_found(user_file):
    """Test error for non-existent password file path."""
    with pytest.raises(ValueError) as excinfo:
        RedisCredentials(
            user_path=user_file,
            password_path="non_existent_password_file.txt",  # noqa
        )
    assert "Path does not point to a file" in str(excinfo.value)


def test_redis_credentials_missing_user_path(password_file):
    """Verify error on missing user file path."""
    with pytest.raises(ValidationError) as excinfo:
        RedisCredentials(password_path=password_file)
    assert "user_path\n  Field required" in str(excinfo.value)


def test_redis_credentials_missing_password_path(user_file):
    """Check for error when password file path is missing."""
    with pytest.raises(ValidationError) as excinfo:
        RedisCredentials(user_path=user_file)
    assert "password_path\n  Field required" in str(excinfo.value)


def test_redis_credentials_missing_both_paths():
    """Test validation errors for missing user and password paths."""
    with pytest.raises(ValidationError) as excinfo:
        RedisCredentials()
    assert "user_path\n  Field required" in str(excinfo.value)
    assert "password_path\n  Field required" in str(excinfo.value)


@pytest.fixture
def mock_redis_credentials(user_file, password_file):
    """Provide a mock RedisCredentials instance."""
    return RedisCredentials(user_path=user_file, password_path=password_file)


@pytest.fixture
def valid_redis_config(mock_redis_credentials):
    """Generate a valid RedisConfig configuration."""
    return {
        "host": "localhost",
        "port": 6379,
        "credentials": mock_redis_credentials,
        "max_memory": "2gb",
        "max_memory_policy": "volatile-lru",
    }


def test_redis_config_success(valid_redis_config):
    """Confirm RedisConfig initializes correctly with valid config."""
    config = RedisConfig(**valid_redis_config)
    assert config.port == 6379
    assert config.max_memory_policy == "volatile-lru"


def test_redis_config_invalid_port(valid_redis_config):
    """Check port validation in RedisConfig."""
    valid_redis_config["port"] = 70000
    with pytest.raises(ValueError) as excinfo:
        RedisConfig(**valid_redis_config)
    assert "Port number must be in 1-65535" in str(excinfo.value)


def test_redis_config_invalid_max_memory_policy(valid_redis_config):
    """Test error on invalid Redis max memory policy."""
    valid_redis_config["max_memory_policy"] = "invalid-policy"
    with pytest.raises(ValueError) as excinfo:
        RedisConfig(**valid_redis_config)
    assert "Invalid Redis max_memory_policy" in str(excinfo.value)


def test_redis_config_valid_memory_policy(valid_redis_config):
    """Verify valid Redis memory policies are accepted."""
    for policy in constants.REDIS_CACHE_MAX_MEMORY_POLICIES:
        valid_redis_config["max_memory_policy"] = policy
        config = RedisConfig(**valid_redis_config)
        assert config.max_memory_policy == policy


def test_memory_config_default_max_entries():
    """Check default max entries in MemoryConfig."""
    config = MemoryConfig()
    assert config.max_entries == constants.IN_MEMORY_CACHE_MAX_ENTRIES


def test_memory_config_explicit_max_entries():
    """Test setting explicit max entries value."""
    explicit_value = 5
    config = MemoryConfig(max_entries=explicit_value)
    assert config.max_entries == explicit_value


def test_memory_config_negative_max_entries():
    """Ensure negative max entries value raises error."""
    with pytest.raises(ValidationError) as excinfo:
        MemoryConfig(max_entries=-1)
    assert "Input should be greater than 0" in str(excinfo.value)


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
        **{
            "host": "other_host",
            "port": 1234,
            "dbname": "my_database",
            "user": "admin",
            "password_path": "tests/config/postgres_password.txt",
            "ssl_mode": "allow",
            "max_entries": 42,
        }
    )
    # check if password was read correctly from file
    assert postgres_config.password == "postgres_password"  # noqa: S105


def test_conversation_cache_config_invalid_type():
    """Test error on invalid cache type in ConversationCacheConfig."""
    with pytest.raises(ValidationError) as excinfo:
        ConversationCacheConfig(type="invalid_cache_type")
    assert "Input should be 'redis' or 'memory'" in str(excinfo.value)


def test_logging_config_default_log_levels():
    """Check default log levels in LoggingConfig."""
    config = LoggingConfig()
    assert logging.getLevelName(config.app_log_level) == "INFO"
    assert logging.getLevelName(config.lib_log_level) == "WARNING"


def test_logging_config_valid_custom_log_levels():
    """Test custom log levels validation in LoggingConfig."""
    config = LoggingConfig(app_log_level="debug", lib_log_level="error")
    assert config.app_log_level == logging.DEBUG
    assert config.lib_log_level == logging.ERROR


def test_logging_config_invalid_log_levels():
    """Ensure invalid log levels raise error in LoggingConfig."""
    with pytest.raises(ValueError) as excinfo:
        LoggingConfig(app_log_level="invalid_level")
    assert "invalid_level is not a valid log level" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        LoggingConfig(lib_log_level="no_level")
    assert "no_level is not a valid log level" in str(excinfo.value)


def test_logging_config_log_level_name_to_numeric_conversion():
    """Verify log level name to numeric conversion in LoggingConfig."""
    config = LoggingConfig(app_log_level="info", lib_log_level="warning")
    assert isinstance(config.app_log_level, int)
    assert isinstance(config.lib_log_level, int)


def test_reference_content_success():
    """Test successful ReferenceContent initialization."""
    with TemporaryDirectory() as tempdir:
        config = ReferenceContent(
            product_docs_index_path=tempdir,
            product_docs_index_id="valid_id",
            embeddings_model_path=tempdir,
        )
        assert config.product_docs_index_path == Path(tempdir)
        assert config.product_docs_index_id == "valid_id"


def test_reference_content_invalid_directory_path():
    """Verify error on invalid directory path in ReferenceContent."""
    invalid_path = "/path/to/nonexistent/directory"
    with pytest.raises(ValidationError) as excinfo:
        ReferenceContent(
            product_docs_index_path=invalid_path,
            product_docs_index_id="valid_id",
            embeddings_model_path=invalid_path,
        )
    assert "Path does not point to a directory" in str(excinfo.value)


@pytest.fixture
def conversation_cache_config():
    """Provide a mock ConversationCacheConfig with Redis type."""
    return ConversationCacheConfig(type="redis")


@pytest.fixture
def logging_config():
    """Generate a LoggingConfig with predefined log levels."""
    return LoggingConfig(app_log_level="info", lib_log_level="warning")


@pytest.fixture
def reference_content():
    """Create a temporary directory for ReferenceContent fixture."""
    with TemporaryDirectory() as tempdir:
        return ReferenceContent(
            product_docs_index_path=tempdir,
            product_docs_index_id="123",
            embeddings_model_path=tempdir,
        )


def test_ols_config_with_all_fields_provided(
    conversation_cache_config, logging_config, reference_content
):
    """Ensure OLSConfig correctly incorporates all provided configurations."""
    ols_config = OLSConfig(
        conversation_cache=conversation_cache_config,
        logging_config=logging_config,
        reference_content=reference_content,
        default_provider="provider",
        default_model="model",
        query_validation_method="disabled",
    )
    assert ols_config.conversation_cache == conversation_cache_config
    assert ols_config.logging_config == logging_config
    assert ols_config.reference_content == reference_content
    assert ols_config.default_provider == "provider"
    assert ols_config.default_model == "model"


def test_ols_config_with_required_fields_only(
    conversation_cache_config, logging_config
):
    """Verify OLSConfig initialization with only required fields."""
    ols_config = OLSConfig(
        conversation_cache=conversation_cache_config, logging_config=logging_config
    )
    assert ols_config.query_validation_method == "llm"
    assert ols_config.conversation_cache == conversation_cache_config
    assert ols_config.logging_config == logging_config
    assert ols_config.reference_content is None
    assert ols_config.default_provider is None
    assert ols_config.default_model is None


def test_ols_config_with_optional_fields_provided(
    conversation_cache_config, logging_config
):
    """Check OLSConfig handles optional fields correctly."""
    ols_config = OLSConfig(
        conversation_cache=conversation_cache_config,
        logging_config=logging_config,
        default_provider="optional_provider",
        default_model="optional_model",
    )
    assert ols_config.default_provider == "optional_provider"
    assert ols_config.default_model == "optional_model"


def test_dev_config_default_values():
    """Verify default values of DevConfig are correctly set."""
    config = DevConfig()
    assert config.enable_dev_ui is False
    assert config.disable_question_validation is False
    assert config.llm_params is None
    assert config.disable_auth is False


@pytest.fixture
def valid_llm_provider_config(credentials_file):
    """Provide a valid LLMProviderConfig instance."""
    return LLMProviderConfig(
        url="http://example.com",
        credentials_path=credentials_file,
        models=[],
        type="bam",
    )


@pytest.fixture
def valid_llm_provider_config_2(credentials_file):
    """Provide another valid LLMProviderConfig with a model config."""
    return LLMProviderConfig(
        url="http://example.com",
        credentials_path=credentials_file,
        models=[ModelConfig(name="test1")],
        type="bam",
    )


@pytest.fixture
def valid_ols_config(conversation_cache_config, logging_config):
    """Generate a valid OLSConfig instance."""
    return OLSConfig(
        conversation_cache=conversation_cache_config, logging_config=logging_config
    )


@pytest.fixture
def valid_ols_config_2(conversation_cache_config, logging_config):
    """Generate an OLSConfig with an invalid default provider for testing."""
    return OLSConfig(
        conversation_cache=conversation_cache_config,
        logging_config=logging_config,
        default_provider="whatever-wrong",
        default_model="test1",
    )


@pytest.fixture
def valid_ols_config_3(conversation_cache_config, logging_config):
    """Provide an OLSConfig with a valid default provider and model."""
    return OLSConfig(
        conversation_cache=conversation_cache_config,
        logging_config=logging_config,
        default_provider="provider1",
        default_model="test1",
    )


@pytest.fixture
def valid_ols_config_4(conversation_cache_config, logging_config):
    """Provide an OLSConfig with a valid provider but invalid default model."""
    return OLSConfig(
        conversation_cache=conversation_cache_config,
        logging_config=logging_config,
        default_provider="provider1",
        default_model="test-wrong",
    )


@pytest.fixture
def valid_dev_config():
    """Generate a DevConfig with dev UI enabled."""
    return DevConfig(enable_dev_ui=True)


def test_config_with_valid_configs(valid_llm_provider_config, valid_ols_config):
    """Test Config initialization with valid LLMProvider and OLSConfig."""
    config = Config(
        llm_providers={"provider1": valid_llm_provider_config},
        ols_config=valid_ols_config,
    )
    print(config)
    assert config.llm_providers["provider1"] == valid_llm_provider_config
    assert config.ols_config == valid_ols_config
    assert config.dev_config is None


def test_config_with_valid_configs_invalid_default_provider(
    valid_llm_provider_config, valid_ols_config_2
):
    """Test Config initialization fails with invalid default provider reference."""
    with pytest.raises(ValidationError) as excinfo:
        Config(
            llm_providers={"provider1": valid_llm_provider_config},
            ols_config=valid_ols_config_2,
        )
    assert "'whatever-wrong' is not one of 'llm_providers'" in str(excinfo.value)


def test_config_with_valid_configs_valid_default_provider(
    valid_llm_provider_config_2, valid_ols_config_3
):
    """Verify Config initialization with a valid default provider."""
    config = Config(
        llm_providers={"provider1": valid_llm_provider_config_2},
        ols_config=valid_ols_config_3,
    )
    assert config.llm_providers["provider1"] == valid_llm_provider_config_2
    assert config.ols_config == valid_ols_config_3


def test_config_with_valid_configs_invalid_default_model(
    valid_llm_provider_config_2, valid_ols_config_4
):
    """Check Config initialization fails with invalid default model reference."""
    with pytest.raises(ValidationError) as excinfo:
        Config(
            llm_providers={"provider1": valid_llm_provider_config_2},
            ols_config=valid_ols_config_4,
        )
    assert "'test-wrong' is not in the models list for provider 'provider1'" in str(
        excinfo.value
    )


def test_config_with_optional_dev_config(
    valid_llm_provider_config, valid_ols_config, valid_dev_config
):
    """Test Config initialization with an optional DevConfig."""
    config = Config(
        llm_providers={"provider1": valid_llm_provider_config},
        ols_config=valid_ols_config,
        dev_config=valid_dev_config,
    )
    assert config.dev_config == valid_dev_config


def test_config_with_invalid_llm_providers(valid_ols_config):
    """Ensure Config initialization fails with invalid LLMProviderConfig instances."""
    with pytest.raises(Exception) as excinfo:
        Config(
            llm_providers={"invalid_provider": "not_a_LLMProviderConfig_instance"},
            ols_config=valid_ols_config,
        )
    assert "Input should be a valid dictionary or instance of LLMProviderConfig" in str(
        excinfo.value
    )
