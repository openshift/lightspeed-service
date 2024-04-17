"""Config classes for the configuration structure."""

import logging
import os
import re
from typing import Any, Optional, Self
from urllib.parse import urlparse

from pydantic import BaseModel, PositiveInt, model_validator

from ols import constants


def _is_valid_http_url(url: str) -> bool:
    """Check is a string is a well-formed http or https URL."""
    result = urlparse(url)
    return all([result.scheme, result.netloc]) and result.scheme in {
        "http",
        "https",
    }


def _get_attribute_from_file(data: dict, file_name_key: str) -> Optional[str]:
    """Retrieve value of an attribute from a file."""
    file_path = data.get(file_name_key)
    if file_path is not None:
        with open(file_path, mode="r", encoding="utf-8") as f:
            return f.read().rstrip()
    return None


def _dir_check(path: str, desc: str) -> None:
    """Check that path is a readable directory."""
    if not os.path.exists(path):
        raise InvalidConfigurationError(f"{desc} '{path}' does not exist")
    if not os.path.isdir(path):
        raise InvalidConfigurationError(f"{desc} '{path}' is not a directory")
    if not os.access(path, os.R_OK):
        raise InvalidConfigurationError(f"{desc} '{path}' is not readable")


def _file_check(path: str, desc: str) -> None:
    """Check that path is a readable regular file."""
    if not os.path.isfile(path):
        raise InvalidConfigurationError(f"{desc} '{path}' is not a file")
    if not os.access(path, os.R_OK):
        raise InvalidConfigurationError(f"{desc} '{path}' is not readable")


class InvalidConfigurationError(Exception):
    """OLS Configuration is invalid."""


class ModelConfig(BaseModel):
    """Model configuration."""

    name: Optional[str] = None
    url: Optional[str] = None
    credentials: Optional[str] = None
    context_window_size: int = constants.DEFAULT_CONTEXT_WINDOW_SIZE
    response_token_limit: int = constants.DEFAULT_RESPONSE_TOKEN_LIMIT
    options: Optional[dict[str, Any]] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.name = data.get("name", None)
        self.url = data.get("url", None)
        self.credentials = _get_attribute_from_file(data, "credentials_path")
        self.context_window_size = self._validate_token_limit(
            data, "context_window_size", self.context_window_size
        )
        self.response_token_limit = self._validate_token_limit(
            data, "response_token_limit", self.response_token_limit
        )
        if self.context_window_size <= self.response_token_limit:
            raise InvalidConfigurationError(
                f"Context window size {self.context_window_size}, "
                f"should be greater than response token limit {self.response_token_limit}"
            )
        # fully optional model-specific options
        self.options = data.get("options", None)

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, ModelConfig):
            return (
                self.name == other.name
                and self.url == other.url
                and self.credentials == other.credentials
                and self.context_window_size == other.context_window_size
                and self.response_token_limit == other.response_token_limit
                and self.options == other.options
            )
        return False

    @staticmethod
    def _validate_token_limit(data: dict, token_type: str, value: int) -> int:
        """Validate token limit."""
        if token_type in data:
            value = data[token_type]
            try:
                value = int(value)
                if value <= 0:
                    raise ValueError
            except (ValueError, TypeError):
                raise InvalidConfigurationError(
                    f"invalid {token_type} = {value}, positive value expected"
                )
        return value

    @staticmethod
    def _validate_model_options(options: dict) -> None:
        """Validate model options which must be dict[str, Any]."""
        if not isinstance(options, dict):
            raise InvalidConfigurationError("model options must be dictionary")
        for key in options:
            if not isinstance(key, str):
                raise InvalidConfigurationError("key for model option must be string")

    def validate_yaml(self) -> None:
        """Validate model config."""
        if self.name is None:
            raise InvalidConfigurationError("model name is missing")
        if self.url is not None and not _is_valid_http_url(self.url):
            raise InvalidConfigurationError(
                "model URL is invalid, only http:// and https:// URLs are supported"
            )
        # model options can be None
        if self.options is not None:
            ModelConfig._validate_model_options(self.options)


class TLSConfig(BaseModel):
    """TLS configuration."""

    tls_certificate_path: Optional[str] = None
    tls_key_path: Optional[str] = None
    tls_key_password: Optional[str] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data:
            self.tls_certificate_path = data.get(
                "tls_certificate_path", self.tls_certificate_path
            )
            self.tls_key_path = data.get("tls_key_path", self.tls_key_path)
            self.tls_key_password = _get_attribute_from_file(
                data, "tls_key_password_path"
            )

    def validate_yaml(self, disable_tls: bool = False) -> None:
        """Validate TLS config."""
        if not disable_tls:
            if not self.tls_certificate_path:
                raise InvalidConfigurationError(
                    "Can not enable TLS without ols_config.tls_config.tls_certificate_path"
                )

            _file_check(self.tls_certificate_path, "OLS server certificate")
            if not self.tls_key_path:
                raise InvalidConfigurationError(
                    "Can not enable TLS without ols_config.tls_config.tls_key_path"
                )
            _file_check(self.tls_key_path, "OLS server certificate private key")


class AuthenticationConfig(BaseModel):
    """Authentication configuration."""

    skip_tls_verification: Optional[bool] = False
    k8s_cluster_api: Optional[str] = None
    k8s_ca_cert_path: Optional[str] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is not None:
            self.skip_tls_verification = data.get(
                "skip_tls_verification", self.skip_tls_verification
            )
            self.k8s_cluster_api = data.get("k8s_cluster_api", self.k8s_cluster_api)
            self.k8s_ca_cert_path = data.get("k8s_ca_cert_path", self.k8s_ca_cert_path)

    def validate_yaml(self) -> None:
        """Validate authentication config."""
        if self.k8s_cluster_api and not _is_valid_http_url(self.k8s_cluster_api):
            raise InvalidConfigurationError("k8s_cluster_api URL is invalid")
        # Validate k8s_ca_cert_path
        if self.k8s_ca_cert_path:
            if not os.path.exists(self.k8s_ca_cert_path):
                raise InvalidConfigurationError(
                    f"k8s_ca_cert_path does not exist: {self.k8s_ca_cert_path}"
                )
            if not os.path.isfile(self.k8s_ca_cert_path):
                raise InvalidConfigurationError(
                    f"k8s_ca_cert_path is not a file: {self.k8s_ca_cert_path}"
                )


class ProviderConfig(BaseModel):
    """LLM provider configuration."""

    name: Optional[str] = None
    type: Optional[str] = None
    url: Optional[str] = None
    credentials: Optional[str] = None
    project_id: Optional[str] = None
    models: dict[str, ModelConfig] = {}
    deployment_name: Optional[str] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.name = data.get("name", None)
        # Default provider type to be the provider name, unless
        # specified explicitly.
        self.type = str(data.get("type", self.name)).lower()
        if self.type not in constants.SUPPORTED_PROVIDER_TYPES:
            raise InvalidConfigurationError(
                f"invalid provider type: {self.type}, supported types are"
                f" {set(constants.SUPPORTED_PROVIDER_TYPES)}"
            )
        self.url = data.get("url", None)
        self.credentials = _get_attribute_from_file(data, "credentials_path")
        self.project_id = data.get("project_id", None)
        if self.type == constants.PROVIDER_WATSONX and self.project_id is None:
            raise InvalidConfigurationError(
                f"project_id is required for WatsonX provider {self.name}"
            )

        if "models" not in data or len(data["models"]) == 0:
            raise InvalidConfigurationError(
                f"no models configured for provider {data['name']}"
            )
        for m in data["models"]:
            if "name" not in m:
                raise InvalidConfigurationError("model name is missing")
            model = ModelConfig(m)
            self.models[m["name"]] = model
        if self.type == constants.PROVIDER_AZURE_OPENAI:
            # deployment_name only required when using Azure OpenAI
            self.deployment_name = data.get("deployment_name", None)
            if self.deployment_name is None:
                raise InvalidConfigurationError(
                    f"deployment_name is required for Azure OpenAI provider {self.name}"
                )

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, ProviderConfig):
            return (
                self.name == other.name
                and self.type == other.type
                and self.url == other.url
                and self.credentials == other.credentials
                and self.project_id == other.project_id
                and self.models == other.models
            )
        return False

    def validate_yaml(self) -> None:
        """Validate provider config."""
        if self.name is None:
            raise InvalidConfigurationError("provider name is missing")
        if self.url is not None and not _is_valid_http_url(self.url):
            raise InvalidConfigurationError(
                "provider URL is invalid, only http:// and https:// URLs are supported"
            )
        for v in self.models.values():
            v.validate_yaml()


class LLMProviders(BaseModel):
    """LLM providers configuration."""

    providers: dict[str, ProviderConfig] = {}

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        for p in data:
            if "name" not in p:
                raise InvalidConfigurationError("provider name is missing")
            provider = ProviderConfig(p)
            self.providers[p["name"]] = provider

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, LLMProviders):
            return self.providers == other.providers
        return False

    def validate_yaml(self) -> None:
        """Validate LLM config."""
        for v in self.providers.values():
            v.validate_yaml()


class PostgresConfig(BaseModel):
    """Postgres configuration."""

    host: str = constants.POSTGRES_CACHE_HOST
    port: PositiveInt = constants.POSTGRES_CACHE_PORT
    dbname: str = constants.POSTGRES_CACHE_DBNAME
    user: str = constants.POSTGRES_CACHE_USER
    password_path: Optional[str] = None
    password: Optional[str] = None
    require_ssl: bool = False
    ca_cert_path: Optional[str] = None
    max_entries: PositiveInt = constants.POSTGRES_CACHE_MAX_ENTRIES

    def __init__(self, **data: Any) -> None:
        """Initialize configuration."""
        super().__init__(**data)
        # password should be read from file
        if self.password_path is not None:
            with open(self.password_path) as f:
                self.password = f.read().rstrip()

    @model_validator(mode="after")
    def validate_yaml(self) -> Self:
        """Validate Postgres cache config."""
        if not (0 < self.port < 65536):
            raise ValueError("The port needs to be between 0 and 65536")
        return self


class RedisConfig(BaseModel):
    """Redis configuration."""

    host: Optional[str] = None
    port: Optional[int] = None
    max_memory: Optional[str] = None
    max_memory_policy: Optional[str] = None
    password: Optional[str] = None
    ca_cert_path: Optional[str] = None
    retry_on_error: Optional[bool] = None
    retry_on_timeout: Optional[bool] = None
    number_of_retries: Optional[int] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.host = data.get("host", constants.REDIS_CACHE_HOST)

        try:
            yaml_port = data.get("port", constants.REDIS_CACHE_PORT)
            self.port = int(yaml_port)
            if not (0 < self.port < 65536):
                raise ValueError
        except ValueError:
            raise InvalidConfigurationError(
                f"invalid Redis port {yaml_port}, valid ports are integers in the (0, 65536) range"
            )

        self.max_memory = data.get("max_memory", constants.REDIS_CACHE_MAX_MEMORY)

        self.max_memory_policy = data.get(
            "max_memory_policy", constants.REDIS_CACHE_MAX_MEMORY_POLICY
        )
        self.ca_cert_path = data.get("ca_cert_path", None)
        self.password = _get_attribute_from_file(data, "password_path")
        self.retry_on_error = (
            str(data.get("retry_on_error", constants.REDIS_RETRY_ON_ERROR)).lower()
            == "true"
        )
        self.retry_on_timeout = (
            str(data.get("retry_on_timeout", constants.REDIS_RETRY_ON_TIMEOUT)).lower()
            == "true"
        )
        self.number_of_retries = int(
            data.get("number_of_retries", constants.REDIS_NUMBER_OF_RETRIES)
        )

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, RedisConfig):
            return (
                self.host == other.host
                and self.port == other.port
                and self.max_memory == other.max_memory
                and self.max_memory_policy == other.max_memory_policy
                and self.password == other.password
                and self.ca_cert_path == other.ca_cert_path
                and self.retry_on_error == other.retry_on_error
                and self.retry_on_timeout == other.retry_on_timeout
                and self.number_of_retries == other.number_of_retries
            )
        return False

    def validate_yaml(self) -> None:
        """Validate Redis cache config."""
        if (
            self.max_memory_policy is not None
            and self.max_memory_policy not in constants.REDIS_CACHE_MAX_MEMORY_POLICIES
        ):
            valid_polices = ", ".join(
                str(p) for p in constants.REDIS_CACHE_MAX_MEMORY_POLICIES
            )
            raise InvalidConfigurationError(
                f"invalid Redis max_memory_policy {self.max_memory_policy},"
                f" valid policies are ({valid_polices})"
            )


class MemoryConfig(BaseModel):
    """In-memory cache configuration."""

    max_entries: Optional[int] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return

        try:
            self.max_entries = int(
                data.get("max_entries", constants.IN_MEMORY_CACHE_MAX_ENTRIES)
            )
            if self.max_entries < 0:
                raise ValueError
        except ValueError:
            raise InvalidConfigurationError(
                "invalid max_entries for memory conversation cache,"
                " max_entries needs to be a non-negative integer"
            )

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, MemoryConfig):
            return self.max_entries == other.max_entries
        return False

    def validate_yaml(self) -> None:
        """Validate memory cache config."""


class QueryFilter(BaseModel):
    """QueryFilter configuration."""

    name: Optional[str] = None
    pattern: Optional[str] = None
    replace_with: Optional[str] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        try:
            self.name = data.get("name")
            self.pattern = data.get("pattern")
            self.replace_with = data.get("replace_with")
            if self.name is None or self.pattern is None or self.replace_with is None:
                raise ValueError
        except ValueError:
            raise InvalidConfigurationError(
                "name, pattern and replace_with need to be specified"
            )

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, QueryFilter):
            return (
                self.name == other.name
                and self.pattern == other.pattern
                and self.replace_with == other.replace_with
            )
        return False

    def validate_yaml(self) -> None:
        """Validate memory cache config."""
        if self.name is None:
            raise InvalidConfigurationError("name is missing")
        if self.pattern is None:
            raise InvalidConfigurationError("pattern is missing")
        try:
            re.compile(self.pattern)
        except re.error:
            raise InvalidConfigurationError("pattern is invalid")
        if self.replace_with is None:
            raise InvalidConfigurationError("replace_with is missing")


class ConversationCacheConfig(BaseModel):
    """Conversation cache configuration."""

    type: Optional[str] = None
    redis: Optional[RedisConfig] = None
    memory: Optional[MemoryConfig] = None
    postgres: Optional[PostgresConfig] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.type = data.get("type", None)
        if self.type is not None:
            match self.type:
                case constants.REDIS_CACHE:
                    if constants.REDIS_CACHE not in data:
                        raise InvalidConfigurationError(
                            "redis conversation cache type is specified,"
                            " but redis configuration is missing"
                        )
                    self.redis = RedisConfig(data.get(constants.REDIS_CACHE))
                case constants.IN_MEMORY_CACHE:
                    if constants.IN_MEMORY_CACHE not in data:
                        raise InvalidConfigurationError(
                            "memory conversation cache type is specified,"
                            " but memory configuration is missing"
                        )
                    self.memory = MemoryConfig(data.get(constants.IN_MEMORY_CACHE))
                case constants.POSTGRES_CACHE:
                    if constants.POSTGRES_CACHE not in data:
                        raise InvalidConfigurationError(
                            "Postgres conversation cache type is specified,"
                            " but Postgres configuration is missing"
                        )
                    self.postgres = PostgresConfig(**data.get(constants.POSTGRES_CACHE))
                case _:
                    raise InvalidConfigurationError(
                        f"unknown conversation cache type: {self.type}"
                    )

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, ConversationCacheConfig):
            return (
                self.type == other.type
                and self.redis == other.redis
                and self.memory == other.memory
            )
        return False

    def validate_yaml(self) -> None:
        """Validate conversation cache config."""
        if self.type is None:
            raise InvalidConfigurationError("missing conversation cache type")
        # cache type is specified, we can decide which cache configuration to validate
        match self.type:
            case constants.REDIS_CACHE:
                self.redis.validate_yaml()
            case constants.IN_MEMORY_CACHE:
                self.memory.validate_yaml()
            case constants.POSTGRES_CACHE:
                pass  # it is validated by Pydantic already
            case _:
                raise InvalidConfigurationError(
                    f"unknown conversation cache type: {self.type}"
                )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    app_log_level: Optional[int] = None
    lib_log_level: Optional[int] = None
    uvicorn_log_level: Optional[int] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            data = {}

        self.app_log_level = self._get_log_level(data, "app_log_level", "info")
        self.lib_log_level = self._get_log_level(data, "lib_log_level", "warning")
        self.uvicorn_log_level = self._get_log_level(
            data, "uvicorn_log_level", "warning"
        )

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, LoggingConfig):
            return (
                self.app_log_level == other.app_log_level
                and self.lib_log_level == other.lib_log_level
                and self.uvicorn_log_level == other.uvicorn_log_level
            )
        return False

    def _get_log_level(self, data: dict, key: str, default: str) -> int:
        log_level = data.get(key, default)
        if not isinstance(log_level, str):
            raise InvalidConfigurationError(f"invalid log level for {log_level}")
        log_level = logging.getLevelName(log_level.upper())
        if not isinstance(log_level, int):
            raise InvalidConfigurationError(
                f"invalid log level for {key}: {data.get(key)}"
            )
        return log_level

    def validate_yaml(self) -> None:
        """Validate logger config."""


class ReferenceContent(BaseModel):
    """Reference content configuration."""

    product_docs_index_path: Optional[str] = None
    product_docs_index_id: Optional[str] = None
    embeddings_model_path: Optional[str] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return

        self.product_docs_index_path = data.get("product_docs_index_path", None)
        self.product_docs_index_id = data.get("product_docs_index_id", None)
        self.embeddings_model_path = data.get("embeddings_model_path", None)

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, ReferenceContent):
            return (
                self.product_docs_index_path == other.product_docs_index_path
                and self.product_docs_index_id == other.product_docs_index_id
                and self.embeddings_model_path == other.embeddings_model_path
            )
        return False

    def validate_yaml(self) -> None:
        """Validate reference content config."""
        if self.product_docs_index_path is not None:
            _dir_check(self.product_docs_index_path, "Reference content path")

            if self.product_docs_index_id is None:
                raise InvalidConfigurationError(
                    "product_docs_index_path is specified but product_docs_index_id is missing"
                )

        if (
            self.product_docs_index_id is not None
            and self.product_docs_index_path is None
        ):
            raise InvalidConfigurationError(
                "product_docs_index_id is specified but product_docs_index_path is missing"
            )

        if self.embeddings_model_path is not None:
            _dir_check(self.embeddings_model_path, "Embeddings model path")


class UserDataCollection(BaseModel):
    """User data collection configuration."""

    feedback_disabled: bool = True
    feedback_storage: Optional[str] = None
    transcripts_disabled: bool = True
    transcripts_storage: Optional[str] = None

    @model_validator(mode="after")
    def check_storage_location_is_set_when_needed(self) -> Self:
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

    conversation_cache: Optional[ConversationCacheConfig] = None
    logging_config: Optional[LoggingConfig] = None
    reference_content: Optional[ReferenceContent] = None
    authentication_config: Optional[AuthenticationConfig] = None
    tls_config: Optional[TLSConfig] = None

    default_provider: Optional[str] = None
    default_model: Optional[str] = None
    query_filters: Optional[list[QueryFilter]] = None
    query_validation_method: Optional[str] = constants.QueryValidationMethod.LLM

    user_data_collection: Optional[UserDataCollection] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return

        self.conversation_cache = ConversationCacheConfig(
            data.get("conversation_cache", None)
        )
        self.logging_config = LoggingConfig(data.get("logging_config", None))
        self.reference_content = ReferenceContent(data.get("reference_content", None))
        self.default_provider = data.get("default_provider", None)
        self.default_model = data.get("default_model", None)
        self.authentication_config = AuthenticationConfig(
            data.get("authentication_config", None)
        )
        self.tls_config = TLSConfig(data.get("tls_config", None))
        if data.get("query_filters", None) is not None:
            self.query_filters = []
            for item in data.get("query_filters", None):
                self.query_filters.append(QueryFilter(item))
        self.query_validation_method = data.get(
            "query_validation_method", constants.QueryValidationMethod.LLM
        )
        self.user_data_collection = UserDataCollection(
            **data.get("user_data_collection", {})
        )

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, OLSConfig):
            return (
                self.conversation_cache == other.conversation_cache
                and self.logging_config == other.logging_config
                and self.reference_content == other.reference_content
                and self.default_provider == other.default_provider
                and self.default_model == other.default_model
                and self.query_filters == other.query_filters
                and self.query_validation_method == other.query_validation_method
                and self.tls_config == other.tls_config
            )
        return False

    def validate_yaml(self, disable_tls: bool = False) -> None:
        """Validate OLS config."""
        self.conversation_cache.validate_yaml()
        self.logging_config.validate_yaml()
        if self.reference_content is not None:
            self.reference_content.validate_yaml()
        if self.authentication_config:
            self.authentication_config.validate_yaml()
        if self.tls_config:
            self.tls_config.validate_yaml(disable_tls)
        if self.query_filters is not None:
            for filter in self.query_filters:
                filter.validate_yaml()

        valid_query_validation_methods = list(constants.QueryValidationMethod)
        if self.query_validation_method not in valid_query_validation_methods:
            raise InvalidConfigurationError(
                f"Invalid query validation method: {self.query_validation_method}\n"
                f"Available options are {valid_query_validation_methods}"
            )


class DevConfig(BaseModel):
    """Developer-mode-only configuration options."""

    enable_dev_ui: bool = False
    llm_params: Optional[dict] = None
    disable_auth: bool = False
    disable_tls: bool = False
    k8s_auth_token: Optional[str] = None
    run_on_localhost: Optional[bool] = False

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize developer configuration settings."""
        super().__init__()
        if data is None:
            return
        self.enable_dev_ui = str(data.get("enable_dev_ui", "False")).lower() == "true"
        self.llm_params = data.get("llm_params", {})
        self.k8s_auth_token = str(data.get("k8s_auth_token", None))
        self.disable_auth = str(data.get("disable_auth", "False")).lower() == "true"
        self.disable_tls = str(data.get("disable_tls", "False")).lower() == "true"
        self.run_on_localhost = (
            str(data.get("run_on_localhost", "False")).lower() == "true"
        )

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, DevConfig):
            return (
                self.enable_dev_ui == other.enable_dev_ui
                and self.llm_params == other.llm_params
                and self.disable_auth == other.disable_auth
                and self.k8s_auth_token == other.k8s_auth_token
                and self.disable_tls == other.disable_tls
                and self.run_on_localhost == other.run_on_localhost
            )
        return False

    def validate_yaml(self) -> None:
        """Validate OLS Dev config."""
        if self.llm_params is not None and not isinstance(self.llm_params, dict):
            raise InvalidConfigurationError("llm_params needs to be defined as a dict")


class Config(BaseModel):
    """Global service configuration."""

    llm_providers: Optional[LLMProviders] = None
    ols_config: Optional[OLSConfig] = None
    dev_config: Optional[DevConfig] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        v = data.get("llm_providers")
        if v is not None:
            self.llm_providers = LLMProviders(v)
        v = data.get("ols_config")
        if v is not None:
            self.ols_config = OLSConfig(v)
        v = data.get("dev_config")
        # Always initialize dev config, even if there's no config for it.
        self.dev_config = DevConfig(v)

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, Config):
            return (
                self.ols_config == other.ols_config
                and self.llm_providers == other.llm_providers
            )
        return False

    def _validate_default_provider_and_model(self) -> None:
        selected_default_provider = self.ols_config.default_provider
        selected_default_model = self.ols_config.default_model

        provider_specified = selected_default_provider is not None
        model_specified = selected_default_model is not None

        if not provider_specified:
            raise InvalidConfigurationError("default_provider is missing")
        if not model_specified:
            raise InvalidConfigurationError("default_model is missing")

        # provider and model are specified
        provider_config = self.llm_providers.providers.get(selected_default_provider)
        if provider_config is None:
            raise InvalidConfigurationError(
                f"default_provider specifies an unknown provider {selected_default_provider}"
            )
        model_config = provider_config.models.get(selected_default_model)
        if model_config is None:
            raise InvalidConfigurationError(
                f"default_model specifies an unknown model {selected_default_model}"
            )

    def validate_yaml(self) -> None:
        """Validate all configurations."""
        if self.llm_providers is None:
            raise InvalidConfigurationError("no LLM providers config section found")
        self.llm_providers.validate_yaml()
        self.dev_config.validate_yaml()
        if self.ols_config is None:
            raise InvalidConfigurationError("no OLS config section found")
        self.ols_config.validate_yaml(self.dev_config.disable_tls)
        self._validate_default_provider_and_model()
