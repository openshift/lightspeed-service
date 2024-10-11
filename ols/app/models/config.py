"""Config classes for the configuration structure."""

import logging
import os
import re
from typing import Any, Literal, Optional, Self
from urllib.parse import urlparse

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


def _is_valid_http_url(url: AnyHttpUrl) -> bool:
    """Check is a string is a well-formed http or https URL."""
    result = urlparse(str(url))
    return all([result.scheme, result.netloc]) and result.scheme in {
        "http",
        "https",
    }


def _get_attribute_from_file(data: dict, file_name_key: str) -> Optional[str]:
    """Retrieve value of an attribute from a file."""
    file_path = data.get(file_name_key)
    if file_path is not None:
        with open(file_path, encoding="utf-8") as f:
            return f.read().rstrip()
    return None


def _read_secret(
    data: dict,
    path_key: str,
    default_filename: str,
    raise_on_error: bool = True,
    directory_name_expected: bool = False,
) -> Optional[str]:
    """Read secret from file on given path or from filename if path points to directory."""
    path = data.get(path_key)

    if path is None:
        return None

    filename = path
    if os.path.isdir(path):
        filename = os.path.join(path, default_filename)
    elif directory_name_expected:
        msg = "Improper credentials_path specified: it must contain path to directory with secrets."
        # no logging configured yet
        print(msg)
        return None

    try:
        with open(filename, encoding="utf-8") as f:
            return f.read().rstrip()
    except OSError as e:
        # some files with secret must exist, so for such cases it is time
        # to inform about improper configuration
        if raise_on_error:
            raise
        # no logging configured yet
        print(f"Problem reading secret from file {filename}:", e)
        print(f"Verify the provider secret contains {default_filename}")
        return None


def _dir_check(path: FilePath, desc: str) -> None:
    """Check that path is a readable directory."""
    if not os.path.exists(path):
        raise InvalidConfigurationError(f"{desc} '{path}' does not exist")
    if not os.path.isdir(path):
        raise InvalidConfigurationError(f"{desc} '{path}' is not a directory")
    if not os.access(path, os.R_OK):
        raise InvalidConfigurationError(f"{desc} '{path}' is not readable")


def _file_check(path: FilePath, desc: str) -> None:
    """Check that path is a readable regular file."""
    if not os.path.isfile(path):
        raise InvalidConfigurationError(f"{desc} '{path}' is not a file")
    if not os.access(path, os.R_OK):
        raise InvalidConfigurationError(f"{desc} '{path}' is not readable")


def _get_log_level(value: str) -> int:
    """Get log level from string."""
    if not isinstance(value, str):
        raise InvalidConfigurationError(
            f"'{value}' log level must be string, got {type(value)}"
        )
    log_level = logging.getLevelName(value.upper())
    if not isinstance(log_level, int):
        raise InvalidConfigurationError(
            f"'{value}' is not valid log level, valid levels are "
            f"{[k.lower() for k in logging.getLevelNamesMapping()]}"
        )
    return log_level


class InvalidConfigurationError(Exception):
    """OLS Configuration is invalid."""


class ModelParameters(BaseModel):
    """Model parameters."""

    max_tokens_for_response: PositiveInt = constants.DEFAULT_MAX_TOKENS_FOR_RESPONSE


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str

    # TODO: OLS-656 Switch OLS operator to use provider-specific configuration parameters
    url: Optional[AnyHttpUrl] = None
    credentials: Optional[str] = None

    context_window_size: PositiveInt = constants.DEFAULT_CONTEXT_WINDOW_SIZE
    parameters: ModelParameters = ModelParameters()

    options: Optional[dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def validate_inputs(cls, data: Any) -> None:
        """Validate model inputs."""
        if data.get("name") is None:
            raise InvalidConfigurationError("model name is missing")

        data["credentials"] = _read_secret(
            data, constants.CREDENTIALS_PATH_SELECTOR, constants.API_TOKEN_FILENAME
        )

        # if the context window size is not set explicitly, use default value.
        # Note that it is important to set a correct value; default may not be accurate.
        data["context_window_size"] = data.get(
            "context_window_size", constants.DEFAULT_CONTEXT_WINDOW_SIZE
        )
        return data

    @field_validator("options")
    @classmethod
    def validate_options(cls, options: dict) -> dict[str, Any]:
        """Validate model options which must be dict[str, Any]."""
        if not isinstance(options, dict):
            raise InvalidConfigurationError("model options must be dictionary")
        for key in options:
            if not isinstance(key, str):
                raise InvalidConfigurationError("key for model option must be string")
        return options

    @model_validator(mode="after")
    def validate_context_window_and_max_tokens(self) -> Self:
        """Validate context window size and max tokens for response."""
        if self.context_window_size <= self.parameters.max_tokens_for_response:  # type: ignore [operator]
            raise InvalidConfigurationError(
                f"Context window size {self.context_window_size}, "
                "should be greater than max_tokens_for_response "
                f"{self.parameters.max_tokens_for_response}"
            )
        return self


class TLSConfig(BaseModel):
    """TLS configuration."""

    tls_certificate_path: Optional[FilePath] = None
    tls_key_path: Optional[FilePath] = None
    tls_key_password: Optional[str] = None

    def __init__(
        self, data: Optional[dict] = None, ignore_missing_certs: bool = False
    ) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        self._ignore_missing_certs = ignore_missing_certs
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
        if not disable_tls and not self._ignore_missing_certs:
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

    skip_tls_verification: bool = False
    k8s_cluster_api: Optional[AnyHttpUrl] = None
    k8s_ca_cert_path: Optional[FilePath] = None


class ProviderSpecificConfig(BaseModel, extra="forbid"):
    """Base class with common provider specific configurations."""

    url: AnyHttpUrl  # required attribute
    token: Optional[Any] = None
    api_key: Optional[str] = None


class OpenAIConfig(ProviderSpecificConfig, extra="forbid"):
    """Configuration specific to OpenAI provider."""

    credentials_path: str  # required attribute


class RHOAIVLLMConfig(ProviderSpecificConfig, extra="forbid"):
    """Configuration specific to RHOAI VLLM provider."""

    credentials_path: str  # required attribute


class RHELAIVLLMConfig(ProviderSpecificConfig, extra="forbid"):
    """Configuration specific to RHEL VLLM provider."""

    credentials_path: str  # required attribute


class AzureOpenAIConfig(ProviderSpecificConfig, extra="forbid"):
    """Configuration specific to Azure OpenAI provider."""

    deployment_name: str  # required attribute
    credentials_path: Optional[str] = None
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret_path: Optional[str] = None
    client_secret: Optional[str] = None


class WatsonxConfig(ProviderSpecificConfig, extra="forbid"):
    """Configuration specific to Watsonx provider."""

    credentials_path: str  # required attribute
    project_id: Optional[str] = None


class BAMConfig(ProviderSpecificConfig, extra="forbid"):
    """Configuration specific to BAM provider."""

    credentials_path: str  # required attribute


class ProviderConfig(BaseModel):
    """LLM provider configuration."""

    name: Optional[str] = None
    type: Optional[str] = None
    url: Optional[AnyHttpUrl] = None
    credentials: Optional[str] = None
    project_id: Optional[str] = None
    models: dict[str, ModelConfig] = {}
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None
    openai_config: Optional[OpenAIConfig] = None
    azure_config: Optional[AzureOpenAIConfig] = None
    watsonx_config: Optional[WatsonxConfig] = None
    bam_config: Optional[BAMConfig] = None
    rhoai_vllm_config: Optional[RHOAIVLLMConfig] = None
    rhelai_vllm_config: Optional[RHELAIVLLMConfig] = None
    certificates_store: Optional[str] = None

    def __init__(
        self,
        data: Optional[dict] = None,
        ignore_llm_secrets: bool = False,
        certificate_directory: str = constants.DEFAULT_CERTIFICATE_DIRECTORY,
    ) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.name = data.get("name", None)

        self.set_provider_type(data)
        self.url = data.get("url", None)
        try:
            self.credentials = _read_secret(
                data, constants.CREDENTIALS_PATH_SELECTOR, constants.API_TOKEN_FILENAME
            )
        except FileNotFoundError:
            if ignore_llm_secrets:
                self.credentials = None
            else:
                raise

        # OLS-622: Provider-specific configuration parameters in olsconfig.yaml
        self.project_id = data.get("project_id", None)
        if self.type == constants.PROVIDER_WATSONX and self.project_id is None:
            raise InvalidConfigurationError(
                f"project_id is required for Watsonx provider {self.name}"
            )

        self.set_provider_specific_configuration(data)

        self.setup_models_config(data)

        if self.type == constants.PROVIDER_AZURE_OPENAI:
            self.api_version = data.get(
                "api_version", constants.DEFAULT_AZURE_API_VERSION
            )
            # deployment_name only required when using Azure OpenAI
            self.deployment_name = data.get("deployment_name", None)
            # note: it can be overwritten in azure_config
        if self.type in (constants.PROVIDER_RHOAI_VLLM, constants.PROVIDER_RHELAI_VLLM):
            self.certificates_store = os.path.join(
                certificate_directory, constants.CERTIFICATE_STORAGE_FILENAME
            )

    def set_provider_type(self, data: dict) -> None:
        """Set the provider type."""
        # Default provider type to be the provider name, unless
        # specified explicitly.
        self.type = str(data.get("type", self.name)).lower()
        if self.type not in constants.SUPPORTED_PROVIDER_TYPES:
            raise InvalidConfigurationError(
                f"invalid provider type: {self.type}, supported types are"
                f" {set(constants.SUPPORTED_PROVIDER_TYPES)}"
            )

    def setup_models_config(self, data: dict) -> None:
        """Set up models configuration."""
        if "models" not in data or len(data["models"]) == 0:
            raise InvalidConfigurationError(
                f"no models configured for provider {data['name']}"
            )
        for m in data["models"]:
            if "name" not in m:
                raise InvalidConfigurationError("model name is missing")
            # add provider to model data - needed for some constants
            # resolution based on the specific provider
            m["provider"] = self.type
            model = ModelConfig(**m)
            self.models[m["name"]] = model

    def set_provider_specific_configuration(self, data: dict) -> None:  # noqa: C901
        """Set the provider-specific configuration."""
        # compute how many provider-specific configurations are
        # found in config file
        found = 0
        for provider_name in constants.SUPPORTED_PROVIDER_TYPES:
            cfg_name = provider_name.lower() + "_config"
            if data.get(cfg_name) is not None:
                found += 1

        # just none or one provider-specific configuration
        # should available
        if found > 1:
            raise InvalidConfigurationError(
                "multiple provider-specific configurations found, "
                f"but just one is expected for provider {self.type}"
            )

        # If one provider-specific configuration is available
        # it must match the selected provider type.
        # It means, that if configuration for selected provider
        # is not present the configuration must be wrong.
        if found == 1:
            match self.type:
                case constants.PROVIDER_AZURE_OPENAI:
                    azure_config = data.get("azure_openai_config")
                    self.check_provider_config(azure_config)
                    if azure_config is not None:
                        azure_config["tenant_id"] = _read_secret(
                            azure_config,
                            constants.CREDENTIALS_PATH_SELECTOR,
                            constants.AZURE_TENANT_ID_FILENAME,
                            directory_name_expected=True,
                            raise_on_error=False,
                        )
                        azure_config["client_id"] = _read_secret(
                            azure_config,
                            constants.CREDENTIALS_PATH_SELECTOR,
                            constants.AZURE_CLIENT_ID_FILENAME,
                            directory_name_expected=True,
                            raise_on_error=False,
                        )
                        azure_config["client_secret"] = _read_secret(
                            azure_config,
                            constants.CREDENTIALS_PATH_SELECTOR,
                            constants.AZURE_CLIENT_SECRET_FILENAME,
                            directory_name_expected=True,
                            raise_on_error=False,
                        )
                    self.read_api_key(azure_config)
                    self.azure_config = AzureOpenAIConfig(**azure_config)
                case constants.PROVIDER_OPENAI:
                    openai_config = data.get("openai_config")
                    self.check_provider_config(openai_config)
                    self.read_api_key(openai_config)
                    self.openai_config = OpenAIConfig(**openai_config)
                case constants.PROVIDER_RHOAI_VLLM:
                    rhoai_vllm_config = data.get("rhoai_vllm_config")
                    self.check_provider_config(rhoai_vllm_config)
                    self.read_api_key(rhoai_vllm_config)
                    self.rhoai_vllm_config = RHOAIVLLMConfig(**rhoai_vllm_config)
                case constants.PROVIDER_RHELAI_VLLM:
                    rhelai_vllm_config = data.get("rhelai_vllm_config")
                    self.check_provider_config(rhelai_vllm_config)
                    self.read_api_key(rhelai_vllm_config)
                    self.rhelai_vllm_config = RHELAIVLLMConfig(**rhelai_vllm_config)
                case constants.PROVIDER_BAM:
                    bam_config = data.get("bam_config")
                    self.check_provider_config(bam_config)
                    self.read_api_key(bam_config)
                    self.bam_config = BAMConfig(**bam_config)
                case constants.PROVIDER_WATSONX:
                    watsonx_config = data.get("watsonx_config")
                    self.check_provider_config(watsonx_config)
                    self.read_api_key(watsonx_config)
                    self.watsonx_config = WatsonxConfig(**watsonx_config)
                case _:
                    raise InvalidConfigurationError(
                        f"Unsupported provider {self.type} configured"
                    )

    @staticmethod
    def read_api_key(config: Optional[dict]) -> None:
        """Read API key from file with secret."""
        if config is None:
            return
        config["api_key"] = _read_secret(
            config,
            constants.CREDENTIALS_PATH_SELECTOR,
            constants.API_TOKEN_FILENAME,
            raise_on_error=False,
        )

    def check_provider_config(self, provider_config: Any) -> None:
        """Check if configuration is presented for selected provider type."""
        if provider_config is None:
            raise InvalidConfigurationError(
                f"provider type {self.type} selected, "
                "but configuration is set for different provider"
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
                and self.azure_config == other.azure_config
                and self.openai_config == other.openai_config
                and self.rhoai_vllm_config == other.rhoai_vllm_config
                and self.rhelai_vllm_config == other.rhelai_vllm_config
                and self.watsonx_config == other.watsonx_config
                and self.bam_config == other.bam_config
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


class LLMProviders(BaseModel):
    """LLM providers configuration."""

    providers: dict[str, ProviderConfig] = {}

    def __init__(
        self,
        data: Optional[dict] = None,
        ignore_llm_secrets: bool = False,
        certificate_directory: str = constants.DEFAULT_CERTIFICATE_DIRECTORY,
    ) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        for p in data:
            if "name" not in p:
                raise InvalidConfigurationError("provider name is missing")
            provider = ProviderConfig(p, ignore_llm_secrets, certificate_directory)
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
    password_path: Optional[FilePath] = None
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
    def validate_yaml(self) -> Self:
        """Validate Postgres cache config."""
        if not 0 < self.port < 65536:
            raise ValueError("The port needs to be between 0 and 65536")
        return self


class RedisConfig(BaseModel):
    """Redis configuration."""

    host: Optional[str] = None
    port: Optional[int] = None
    max_memory: Optional[str] = None
    max_memory_policy: Optional[str] = None
    password: Optional[str] = None
    ca_cert_path: Optional[FilePath] = None
    retry_on_error: Optional[bool] = None
    retry_on_timeout: Optional[bool] = None
    number_of_retries: Optional[int] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.host = data.get("host", constants.REDIS_CACHE_HOST)

        yaml_port = data.get("port", constants.REDIS_CACHE_PORT)
        try:
            self.port = int(yaml_port)
            if not 0 < self.port < 65536:
                raise ValueError
        except ValueError as e:
            raise InvalidConfigurationError(
                f"invalid Redis port {yaml_port}, valid ports are integers in the (0, 65536) range"
            ) from e

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


class InMemoryCacheConfig(BaseModel):
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
        except ValueError as e:
            raise InvalidConfigurationError(
                "invalid max_entries for memory conversation cache,"
                " max_entries needs to be a non-negative integer"
            ) from e

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, InMemoryCacheConfig):
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
        except ValueError as e:
            raise InvalidConfigurationError(
                "name, pattern and replace_with need to be specified"
            ) from e

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
        except re.error as e:
            raise InvalidConfigurationError("pattern is invalid") from e
        if self.replace_with is None:
            raise InvalidConfigurationError("replace_with is missing")


class ConversationCacheConfig(BaseModel):
    """Conversation cache configuration."""

    type: Optional[str] = None
    redis: Optional[RedisConfig] = None
    memory: Optional[InMemoryCacheConfig] = None
    postgres: Optional[PostgresConfig] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.type = data.get("type", None)
        if self.type is not None:
            match self.type:
                case constants.CACHE_TYPE_REDIS:
                    if constants.CACHE_TYPE_REDIS not in data:
                        raise InvalidConfigurationError(
                            "redis conversation cache type is specified,"
                            " but redis configuration is missing"
                        )
                    self.redis = RedisConfig(data.get(constants.CACHE_TYPE_REDIS))
                case constants.CACHE_TYPE_MEMORY:
                    if constants.CACHE_TYPE_MEMORY not in data:
                        raise InvalidConfigurationError(
                            "memory conversation cache type is specified,"
                            " but memory configuration is missing"
                        )
                    self.memory = InMemoryCacheConfig(
                        data.get(constants.CACHE_TYPE_MEMORY)
                    )
                case constants.CACHE_TYPE_POSTGRES:
                    if constants.CACHE_TYPE_POSTGRES not in data:
                        raise InvalidConfigurationError(
                            "Postgres conversation cache type is specified,"
                            " but Postgres configuration is missing"
                        )
                    self.postgres = PostgresConfig(
                        **data.get(constants.CACHE_TYPE_POSTGRES)
                    )
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
                and self.postgres == other.postgres
            )
        return False

    def validate_yaml(self) -> None:
        """Validate conversation cache config."""
        if self.type is None:
            raise InvalidConfigurationError("missing conversation cache type")
        # cache type is specified, we can decide which cache configuration to validate
        match self.type:
            case constants.CACHE_TYPE_REDIS:
                self.redis.validate_yaml()
            case constants.CACHE_TYPE_MEMORY:
                self.memory.validate_yaml()
            case constants.CACHE_TYPE_POSTGRES:
                pass  # it is validated by Pydantic already
            case _:
                raise InvalidConfigurationError(
                    f"unknown conversation cache type: {self.type}"
                )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    app_log_level: int = logging.INFO
    lib_log_level: int = logging.WARNING
    uvicorn_log_level: int = logging.WARNING

    def __init__(self, **data: Optional[dict]) -> None:
        """Initialize configuration and perform basic validation."""
        # convert input strings (level names, eg. debug/info,...) to
        # logging level names (integer values) for defined model fields
        for field in self.model_fields:
            if field in data:
                data[field] = _get_log_level(data[field])  # type: ignore[assignment]
        super().__init__(**data)


class ReferenceContent(BaseModel):
    """Reference content configuration."""

    product_docs_index_path: Optional[FilePath] = None
    product_docs_index_id: Optional[str] = None
    embeddings_model_path: Optional[FilePath] = None

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
    authentication_config: AuthenticationConfig = AuthenticationConfig()
    tls_config: TLSConfig = TLSConfig()
    system_prompt_path: Optional[str] = None
    system_prompt: Optional[str] = None

    default_provider: Optional[str] = None
    default_model: Optional[str] = None
    query_filters: Optional[list[QueryFilter]] = None
    query_validation_method: Optional[str] = constants.QueryValidationMethod.DISABLED

    user_data_collection: UserDataCollection = UserDataCollection()

    extra_ca: list[FilePath] = []
    certificate_directory: Optional[str] = None

    def __init__(
        self, data: Optional[dict] = None, ignore_missing_certs: bool = False
    ) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return

        self.conversation_cache = ConversationCacheConfig(
            data.get("conversation_cache", None)
        )
        self.logging_config = LoggingConfig(**data.get("logging_config", {}))
        if data.get("reference_content") is not None:
            self.reference_content = ReferenceContent(data.get("reference_content"))
        self.default_provider = data.get("default_provider", None)
        self.default_model = data.get("default_model", None)
        self.authentication_config = AuthenticationConfig(
            **data.get("authentication_config", {})
        )
        self.tls_config = TLSConfig(data.get("tls_config", None), ignore_missing_certs)
        if data.get("query_filters", None) is not None:
            self.query_filters = []
            for item in data.get("query_filters", None):
                self.query_filters.append(QueryFilter(item))
        self.query_validation_method = data.get(
            "query_validation_method", constants.QueryValidationMethod.DISABLED
        )
        self.user_data_collection = UserDataCollection(
            **data.get("user_data_collection", {})
        )
        # read file containing system prompt
        # if not specified, the prompt will remain None, which will be handled
        # by system prompt infrastructure
        self.system_prompt = _get_attribute_from_file(data, "system_prompt_path")

        self.extra_ca = data.get("extra_ca", [])
        self.certificate_directory = data.get(
            "certificate_directory", constants.DEFAULT_CERTIFICATE_DIRECTORY
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
                and self.certificate_directory == other.certificate_directory
                and self.system_prompt == other.system_prompt
            )
        return False

    def validate_yaml(self, disable_tls: bool = False) -> None:
        """Validate OLS config."""
        if self.conversation_cache is not None:
            self.conversation_cache.validate_yaml()
        if self.reference_content is not None:
            self.reference_content.validate_yaml()
        if self.tls_config:
            self.tls_config.validate_yaml(disable_tls)
        if self.query_filters is not None:
            for query_filter in self.query_filters:
                query_filter.validate_yaml()

        valid_query_validation_methods = list(constants.QueryValidationMethod)
        if self.query_validation_method not in valid_query_validation_methods:
            raise InvalidConfigurationError(
                f"Invalid query validation method: {self.query_validation_method}\n"
                f"Available options are {valid_query_validation_methods}"
            )


class DevConfig(BaseModel):
    """Developer-mode-only configuration options."""

    enable_dev_ui: bool = False
    llm_params: dict = {}
    disable_auth: bool = False
    disable_tls: bool = False
    k8s_auth_token: Optional[str] = None
    run_on_localhost: bool = False


class UserDataCollectorConfig(BaseModel):
    """User data collection configuration."""

    data_storage: Optional[DirectoryPath] = None
    log_level: int = logging.INFO
    collection_interval: int = 2 * 60 * 60  # 2 hours
    run_without_initial_wait: bool = False
    ingress_env: Literal["stage", "prod"] = "prod"
    cp_offline_token: Optional[str] = None

    def __init__(self, **data: Optional[dict]) -> None:
        """Initialize configuration."""
        # convert input strings (level names, eg. debug/info,...) to
        # logging level name (integer values)
        if "log_level" in data:
            data["log_level"] = _get_log_level(data["log_level"])  # type: ignore[assignment]
        super().__init__(**data)

    @model_validator(mode="after")
    def validate_token_is_set_when_needed(self) -> Self:
        """Check that cp_offline_token is set when env is stage."""
        if self.ingress_env == "stage" and self.cp_offline_token is None:
            raise ValueError("cp_offline_token is required in stage environment")
        return self


class Config(BaseModel):
    """Global service configuration."""

    llm_providers: LLMProviders = LLMProviders()
    ols_config: OLSConfig = OLSConfig()
    dev_config: DevConfig = DevConfig()
    user_data_collector_config: Optional[UserDataCollectorConfig] = None

    def __init__(
        self,
        data: Optional[dict] = None,
        ignore_llm_secrets: bool = False,
        ignore_missing_certs: bool = False,
    ) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        v = data.get("ols_config")
        if v is not None:
            self.ols_config = OLSConfig(v, ignore_missing_certs)
        else:
            raise InvalidConfigurationError("no OLS config section found")
        v = data.get("llm_providers")
        if v is not None:
            self.llm_providers = LLMProviders(
                v, ignore_llm_secrets, self.ols_config.certificate_directory
            )
        else:
            raise InvalidConfigurationError("no LLM providers config section found")
        # Always initialize dev config, even if there's no config for it.
        self.dev_config = DevConfig(**data.get("dev_config", {}))
        self.user_data_collector_config = UserDataCollectorConfig(
            **data.get("user_data_collector_config", {})
        )

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
        self.llm_providers.validate_yaml()
        self.ols_config.validate_yaml(self.dev_config.disable_tls)
        self._validate_default_provider_and_model()
