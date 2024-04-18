"""Config classes for the configuration structure."""

import logging
import re
from typing import Any, Literal, Optional

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    DirectoryPath,
    FilePath,
    PositiveInt,
    field_validator,
    model_validator,
)

from ols import constants


def _get_attribute_from_file(file_path: str) -> str:
    try:
        with open(file_path, mode="r") as file:
            return file.read().strip()
    except FileNotFoundError:
        raise ValueError(f"File not found at {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading file at {file_path}: {e!s}")


class ModelConfig(BaseModel):
    """Model Configuation."""

    name: str  # duplicates
    url: Optional[AnyHttpUrl] = None
    context_window_size: PositiveInt = constants.DEFAULT_CONTEXT_WINDOW_SIZE
    response_token_limit: PositiveInt = constants.DEFAULT_RESPONSE_TOKEN_LIMIT
    options: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def check_valid_window_size_and_token_limit(self):  # type: ignore
        """Validate window size and token limit."""
        if self.context_window_size <= self.response_token_limit:
            raise ValueError(
                f"Context window size {self.context_window_size}, "
                f"should be greater than response token limit {self.response_token_limit}"
            )
        return self


class TLSConfig(BaseModel):
    """TLS configuration."""

    tls_certificate_path: FilePath
    tls_key_path: FilePath
    tls_key_password: str = None  # type: ignore

    def __init__(self, **data):  # type: ignore
        """Initialize TLS configuration."""
        super().__init__(**data)
        self.tls_key_password = _get_attribute_from_file(self.tls_key_password)


class LLMProviderConfig(BaseModel):
    """LLM Provider Configuration."""

    url: Optional[AnyHttpUrl] = None
    type: str
    credentials_path: FilePath
    project_id: Optional[str] = None
    models: list[ModelConfig]
    credentials: str = None  # type: ignore
    deployment_name: Optional[str] = None

    def __init__(self, **data):  # type: ignore
        """Initialize LLM provider configuration."""
        super().__init__(**data)
        self.credentials = _get_attribute_from_file(self.credentials_path)

    @model_validator(mode="after")
    def check_valid_provider_type(self):  # type: ignore
        """Validate provider type."""
        if self.type not in constants.SUPPORTED_PROVIDER_TYPES:
            raise ValueError(
                f"invalid provider type: {self.type}, supported types are "
                f"{set(constants.SUPPORTED_PROVIDER_TYPES)}"
            )
        if self.type == constants.PROVIDER_WATSONX and not self.project_id:
            raise ValueError(
                f"project_id is required for {constants.PROVIDER_WATSONX} provider"
            )
        return self


class RedisCredentials(BaseModel):
    """Redis credentials."""

    user_path: FilePath
    password_path: FilePath
    username: str = None  # type: ignore
    password: str = None  # type: ignore

    def __init__(self, **data):  # type: ignore
        """Initialize redis credentials."""
        super().__init__(**data)
        self.username = _get_attribute_from_file(self.user_path)
        self.password = _get_attribute_from_file(self.password_path)


class RedisConfig(BaseModel):
    """Redis configuration."""

    host: Optional[str] = constants.REDIS_CACHE_HOST
    port: Optional[int] = constants.REDIS_CACHE_PORT
    max_memory: Optional[str] = constants.REDIS_CACHE_MAX_MEMORY
    max_memory_policy: Optional[str] = constants.REDIS_CACHE_MAX_MEMORY_POLICY
    credentials: Optional[RedisCredentials] = None
    ca_cert_path: Optional[FilePath] = None
    retry_on_error: Optional[bool] = None
    retry_on_timeout: Optional[bool] = None
    number_of_retries: Optional[PositiveInt] = None

    @field_validator("port")
    @classmethod
    def check_valid_tcp_port(cls, v):  # type: ignore
        """Validate tcp port."""
        if not (1 <= v <= 65535):
            raise ValueError("Port number must be in 1-65535")
        return v

    @field_validator("max_memory_policy")
    @classmethod
    def check_valid_memory_policy(cls, v):  # type: ignore
        """Validate max memory policy."""
        if v not in constants.REDIS_CACHE_MAX_MEMORY_POLICIES:
            raise ValueError(
                f"Invalid Redis max_memory_policy: {v}, valid policies are "
                f"({constants.REDIS_CACHE_MAX_MEMORY_POLICIES})"
            )
        return v


class MemoryConfig(BaseModel):
    """In-memory cache configuration."""

    max_entries: PositiveInt = constants.IN_MEMORY_CACHE_MAX_ENTRIES


class PostgresConfig(BaseModel):
    """Postgres configuration."""

    host: str = constants.POSTGRES_CACHE_HOST
    port: PositiveInt = constants.POSTGRES_CACHE_PORT
    dbname: str = constants.POSTGRES_CACHE_DBNAME
    user: str = constants.POSTGRES_CACHE_USER
    password_path: Optional[str] = None
    password: Optional[str] = None
    ssl_mode: str = constants.POSTGRES_CACHE_SSL_MODE
    ca_cert_path: Optional[FilePath] = None
    max_entries: PositiveInt = constants.POSTGRES_CACHE_MAX_ENTRIES

    def __init__(self, **data: Any) -> None:
        """Initialize configuration."""
        super().__init__(**data)
        # password should be read from file
        if self.password_path is not None:
            with open(self.password_path) as f:
                self.password = f.read().rstrip()

    @model_validator(mode="after")
    def validate_yaml(self):  # type: ignore
        """Validate Postgres cache config."""
        if not (0 < self.port < 65536):
            raise ValueError("The port needs to be between 0 and 65536")
        return self


class ConversationCacheConfig(BaseModel):
    """Conversation cache configuration."""

    type: Literal[constants.REDIS_CACHE, constants.IN_MEMORY_CACHE]  # type: ignore
    redis: Optional[RedisConfig] = None
    memory: Optional[MemoryConfig] = None
    postgres: Optional[PostgresConfig] = None

    def __init__(self, **data):  # type: ignore
        """Initialize conversation cache configuration."""
        super().__init__(**data)
        if self.type == constants.REDIS_CACHE:
            if not self.redis:
                self.redis = RedisConfig()
        elif self.type == constants.IN_MEMORY_CACHE:
            if not self.memory:
                self.memory = MemoryConfig()


class LoggingConfig(BaseModel):
    """Logging configuration."""

    app_log_level: Optional[str] = "info"
    lib_log_level: Optional[str] = "warning"
    uvicorn_log_level: Optional[str] = "warning"

    @field_validator("app_log_level", "lib_log_level", "uvicorn_log_level")
    @classmethod
    def validate_log_level(cls, v):  # type: ignore
        """Validate log levels."""
        level = logging.getLevelName(v.upper())

        if not isinstance(level, int):
            raise ValueError(f"{v} is not a valid log level")
        return v

    def __init__(self, **data):  # type: ignore
        """Initialize logging configuration."""
        super().__init__(**data)
        self.app_log_level = logging.getLevelName(self.app_log_level.upper())
        self.lib_log_level = logging.getLevelName(self.lib_log_level.upper())
        self.uvicorn_log_level = logging.getLevelName(self.uvicorn_log_level.upper())


class ReferenceContent(BaseModel):
    """Reference content configuration."""

    product_docs_index_path: DirectoryPath
    product_docs_index_id: str
    embeddings_model_path: DirectoryPath


class QueryFilter(BaseModel):
    """QueryFilter configuration."""

    name: str
    pattern: str
    replace_with: str

    @field_validator("pattern")
    @classmethod
    def check_patern(cls, v: str) -> str:
        """Validate query pattern."""
        try:
            re.compile(v)
        except re.error:
            raise ValueError(f"{v} is not a valid query pattern")
        return v


class AuthenticationConfig(BaseModel):
    """Authentication configuration."""

    skip_tls_verification: Optional[bool] = False
    k8s_cluster_api: AnyHttpUrl
    k8s_ca_cert_path: Optional[FilePath] = None


class UserDataCollection(BaseModel):
    """User data collection configuration."""

    feedback_disabled: bool = True
    feedback_storage: Optional[str] = None
    transcripts_disabled: bool = True
    transcripts_storage: Optional[str] = None

    @model_validator(mode="after")
    def check_storage_location_is_set_when_needed(self):  # type: ignore
        """Check that storage_location is set when enabled."""
        if not self.feedback_disabled and self.feedback_storage is None:
            raise ValueError("feedback_storage is required when feedback is enabled")
        if not self.transcripts_disabled and self.transcripts_storage is None:
            raise ValueError(
                "transcripts_storage is required when transcripts capturing is enabled"
            )
        return self


class OLSConfig(BaseModel):
    """OLS configuration."""

    conversation_cache: ConversationCacheConfig
    logging_config: LoggingConfig = LoggingConfig()  # type: ignore
    reference_content: Optional[ReferenceContent] = None
    authentication_config: Optional[AuthenticationConfig] = None
    tls_config: Optional[TLSConfig] = None

    default_provider: Optional[str] = None
    default_model: Optional[str] = None
    query_filters: Optional[list[QueryFilter]] = None
    query_validation_method: Literal[
        constants.QueryValidationMethod.LLM,
        constants.QueryValidationMethod.KEYWORD,
        constants.QueryValidationMethod.DISABLED,
    ] = constants.QueryValidationMethod.LLM

    user_data_collection: Optional[UserDataCollection] = UserDataCollection()

    @model_validator(mode="before")
    def check_default_provider_and_model_together(self):  # type: ignore
        """Validate default provider and model."""
        default_provider, default_model = self.get("default_provider", None), self.get(
            "default_model", None
        )
        if (default_provider is None) != (default_model is None):
            raise ValueError(
                "Both 'default_provider' and 'default_model' must be provided "
                "together or not at all."
            )

        return self


class DevConfig(BaseModel):
    """Developer-mode-only configuration options."""

    enable_dev_ui: bool = False
    disable_question_validation: bool = False
    llm_params: Optional[dict] = None

    # TODO - wire this up once auth is implemented
    disable_auth: bool = False
    disable_tls: bool = False
    k8s_auth_token: Optional[str] = None
    run_on_localhost: Optional[bool] = False


class Config(BaseModel):
    """Global service configuration."""

    llm_providers: dict[str, LLMProviderConfig]  # duplicate keys
    ols_config: OLSConfig
    dev_config: Optional[DevConfig] = None

    @model_validator(mode="after")
    def check_default_provider_and_model(self):  # type: ignore
        """Validate default provider and model."""
        default_provider = self.ols_config.default_provider
        default_model = self.ols_config.default_model

        if default_provider and default_provider not in self.llm_providers:
            raise ValueError(
                f"default_provider '{default_provider}' is not one of 'llm_providers'"
            )

        if default_provider and default_model:
            provider_config = self.llm_providers[default_provider]
            if provider_config:
                model_names = [model.name for model in provider_config.models]
                if default_model not in model_names:
                    raise ValueError(
                        f"default_model '{default_model}' is not in the models list for provider "
                        f"'{default_provider}'"
                    )

        return self
