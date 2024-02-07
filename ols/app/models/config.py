"""Config classes for the configuration structure."""

import logging
from typing import Any, Literal, Optional
from ols import constants
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    DirectoryPath,
    FilePath,
    PositiveInt,
    field_validator,
    model_validator,
)
import re


def _get_attribute_from_file(file_path):
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
    url: Optional[str] = None
    context_window_size: PositiveInt = constants.DEFAULT_CONTEXT_WINDOW_SIZE
    response_token_limit: PositiveInt = constants.DEFAULT_RESPONSE_TOKEN_LIMIT
    options: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def check_valid_window_size_and_token_limit(cls, v):
        if v.context_window_size <= v.response_token_limit:
            raise ValueError(
                f"Context window size {v.context_window_size}, "
                f"should be greater than response token limit {v.response_token_limit}"
            )
        return v


class TLSConfig(BaseModel):
    """TLS configuration."""

    tls_certificate_path: FilePath
    tls_key_path: FilePath
    tls_key_password: str = None

    def __init__(self, **data):
        """Initialize TLS configuration."""
        super().__init__(**data)
        self.tls_key_password = _get_attribute_from_file(self.tls_key_password)


class LLMProviderConfig(BaseModel):
    """LLM Provider Configuration."""

    url: AnyHttpUrl
    type: str
    credentials_path: FilePath
    project_id: Optional[str] = None
    models: list[ModelConfig]
    credentials: str = None
    deployment_name: Optional[str] = None

    def __init__(self, **data):
        """Initialize LLM provider configuration."""
        super().__init__(**data)
        self.credentials = _get_attribute_from_file(self.credentials_path)

    @model_validator(mode="after")
    def check_valid_provider_type(cls, v):  # noqa: N805
        """Validate provider type."""
        if v.type not in constants.SUPPORTED_PROVIDER_TYPES:
            raise ValueError(
                f"invalid provider type: {type}, supported types are "
                f"{set(constants.SUPPORTED_PROVIDER_TYPES)}"
            )

        if v.type == constants.PROVIDER_WATSONX and not v.project_id:
            raise ValueError(
                f"project_id is required for {constants.PROVIDER_WATSONX} provider"
            )
        return v


class RedisCredentials(BaseModel):
    """Redis credentials."""

    user_path: FilePath
    password_path: FilePath
    username: str = None
    password: str = None

    def __init__(self, **data):
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
    def check_valid_tcp_port(cls, v):  # noqa: N805
        """Validate tcp port."""
        if not (1 <= v <= 65535):
            raise ValueError("Port number must be in 1-65535")
        return v

    @field_validator("max_memory_policy")
    def check_valid_memory_policy(cls, v):  # noqa: N805
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


class ConversationCacheConfig(BaseModel):
    """Conversation cache configuration."""

    type: Literal[constants.REDIS_CACHE, constants.IN_MEMORY_CACHE]
    redis: Optional[RedisConfig] = None
    memory: Optional[MemoryConfig] = None

    def __init__(self, **data):
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

    @field_validator("app_log_level", "lib_log_level")
    def validate_log_level(cls, v):  # noqa: N805
        """Validate log levels."""
        level = logging.getLevelName(v.upper())

        if not isinstance(level, int):
            raise ValueError(f"{v} is not a valid log level")
        return v

    def __init__(self, **data):
        """Initialize logging configuration."""
        super().__init__(**data)
        self.app_log_level = logging.getLevelName(self.app_log_level.upper())
        self.lib_log_level = logging.getLevelName(self.lib_log_level.upper())


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
    def check_patern(cls, v):  # noqa: N805
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
    feedback_storage: Optional[DirectoryPath] = None

    @model_validator(mode="after")
    def check_storage_location_is_set_when_needed(self):
        """Check that storage_location is set when enabled."""
        if not self.feedback_disabled and self.feedback_storage is None:
            raise ValueError("feedback_storage is required when feedback is enabled")
        return self


class OLSConfig(BaseModel):
    """OLS configuration."""

    conversation_cache: ConversationCacheConfig
    logging_config: LoggingConfig = LoggingConfig()
    reference_content: Optional[ReferenceContent] = None
    authentication_config: Optional[AuthenticationConfig] = None
    tls_config: Optional[TLSConfig] = None

    default_provider: Optional[str] = None
    default_model: Optional[str] = None
    query_filters: Optional[list[QueryFilter]] = None

    @model_validator(mode="before")
    def check_default_provider_and_model_together(cls, values):  # noqa: N805
        """Validate default provider and model."""
        default_provider, default_model = values.get("default_provider"), values.get(
            "default_model"
        )
        if (default_provider is None) != (default_model is None):
            raise ValueError(
                "Both 'default_provider' and 'default_model' must be provided "
                "together or not at all."
            )
        return values


class DevConfig(BaseModel):
    """Developer-mode-only configuration options."""

    enable_dev_ui: Optional[bool] = False
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
    def check_default_provider_and_model(cls, v):  # noqa: N805
        """Validate default provider and model."""
        default_provider = v.ols_config.default_provider
        default_model = v.ols_config.default_model

        if default_provider and default_provider not in v.llm_providers:
            raise ValueError(
                f"default_provider '{default_provider}' is not one of 'llm_providers'"
            )

        if default_provider and default_model:
            provider_config = v.llm_providers[default_provider]
            if provider_config:
                model_names = [model.name for model in provider_config.models]
                if default_model not in model_names:
                    raise ValueError(
                        f"default_model '{default_model}' is not in the models list for provider "
                        f"'{default_provider}'"
                    )

        return v
