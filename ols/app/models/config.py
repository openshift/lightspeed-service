"""Config classes for the configuration structure."""

import logging
import os
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel

from ols import constants


def _is_valid_http_url(url: str) -> bool:
    """Check is a string is a well-formed http or https URL."""
    result = urlparse(url)
    return all([result.scheme, result.netloc]) and result.scheme in [
        "http",
        "https",
    ]


def _get_attribute_from_file(data: dict, file_name_key: str) -> Optional[str]:
    """Retrieve value of an attribute from a file."""
    file_path = data.get(file_name_key)
    if file_path is not None:
        with open(file_path, mode="r") as f:
            return f.read().rstrip()
    return None


class InvalidConfigurationError(Exception):
    """OLS Configuration is invalid."""


class ModelConfig(BaseModel):
    """Model configuration."""

    name: Optional[str] = None
    url: Optional[str] = None
    credentials: Optional[str] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.name = data.get("name", None)
        self.url = data.get("url", None)
        self.credentials = _get_attribute_from_file(data, "credentials_path")

    def __eq__(self, other) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, ModelConfig):
            return (
                self.name == other.name
                and self.url == other.url
                and self.credentials == other.credentials
            )
        return False

    def validate_yaml(self) -> None:
        """Validate model config."""
        if self.name is None:
            raise InvalidConfigurationError("model name is missing")
        if self.url is not None and not _is_valid_http_url(self.url):
            raise InvalidConfigurationError("model URL is invalid")


class ProviderConfig(BaseModel):
    """LLM provider configuration."""

    name: Optional[str] = None
    url: Optional[str] = None
    credentials: Optional[str] = None
    models: dict[str, ModelConfig] = {}

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.name = data.get("name", None)
        self.url = data.get("url", None)
        self.credentials = _get_attribute_from_file(data, "credentials_path")
        if "models" not in data or len(data["models"]) == 0:
            raise InvalidConfigurationError(
                f"no models configured for provider {data['name']}"
            )
        for m in data["models"]:
            if "name" not in m:
                raise InvalidConfigurationError("model name is missing")
            model = ModelConfig(m)
            self.models[m["name"]] = model

    def __eq__(self, other) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, ProviderConfig):
            return (
                self.name == other.name
                and self.url == other.url
                and self.credentials == other.credentials
                and self.models == other.models
            )
        return False

    def validate_yaml(self) -> None:
        """Validate provider config."""
        if self.name is None:
            raise InvalidConfigurationError("provider name is missing")
        if self.url is not None and not _is_valid_http_url(self.url):
            raise InvalidConfigurationError("provider URL is invalid")
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

    def __eq__(self, other) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, LLMProviders):
            return self.providers == other.providers
        return False

    def validate_yaml(self) -> None:
        """Validate LLM config."""
        for v in self.providers.values():
            v.validate_yaml()


class RedisCredentials(BaseModel):
    """Redis credentials."""

    user: Optional[str] = None
    password: Optional[str] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize redis credentials."""
        super().__init__()
        if not isinstance(data, dict):
            return
        self.user = _get_attribute_from_file(data, "user_path")
        self.password = _get_attribute_from_file(data, "password_path")

    def __eq__(self, other) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, RedisCredentials):
            return self.user == other.user and self.password == other.password
        return False

    def validate_yaml(self) -> None:
        """Validate redis credentials."""
        if (self.user is not None and self.password is None) or (
            self.user is None and self.password is not None
        ):
            raise InvalidConfigurationError(
                "both or neither user and password need to be specified for Redis"
            )


class RedisConfig(BaseModel):
    """Redis configuration."""

    host: Optional[str] = None
    port: Optional[int] = None
    max_memory: Optional[str] = None
    max_memory_policy: Optional[str] = None
    credentials: Optional[RedisCredentials] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.host = data.get("host", None)
        self.port = data.get("port", None)
        self.max_memory = data.get("max_memory", constants.REDIS_CACHE_MAX_MEMORY)
        self.max_memory_policy = data.get(
            "max_memory_policy", constants.REDIS_CACHE_MAX_MEMORY_POLICY
        )
        credentials = data.get("credentials")
        if credentials is not None:
            self.credentials = RedisCredentials(credentials)

    def __eq__(self, other) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, RedisConfig):
            return (
                self.host == other.host
                and self.port == other.port
                and self.max_memory == other.max_memory
                and self.max_memory_policy == other.max_memory_policy
                and self.credentials == other.credentials
            )
        return False


class MemoryConfig(BaseModel):
    """In-memory cache configuration."""

    max_entries: Optional[int] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.max_entries = data.get(
            "max_entries", constants.IN_MEMORY_CACHE_MAX_ENTRIES
        )

    def __eq__(self, other) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, MemoryConfig):
            return self.max_entries == other.max_entries
        return False


class ConversationCacheConfig(BaseModel):
    """Conversation cache configuration."""

    type: Optional[str] = None
    redis: Optional[RedisConfig] = None
    memory: Optional[MemoryConfig] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.type = data.get("type", None)
        if self.type is not None:
            if self.type == constants.REDIS_CACHE:
                if constants.REDIS_CACHE not in data:
                    raise InvalidConfigurationError("redis configuration is missing")
                self.redis = RedisConfig(data.get(constants.REDIS_CACHE))
            elif self.type == constants.IN_MEMORY_CACHE:
                if constants.IN_MEMORY_CACHE not in data:
                    raise InvalidConfigurationError("memory configuration is missing")
                self.memory = MemoryConfig(data.get(constants.IN_MEMORY_CACHE))
            else:
                raise InvalidConfigurationError("unknown conversation cache store")

    def __eq__(self, other) -> bool:
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


class LoggingConfig(BaseModel):
    """Logging configuration."""

    app_log_level: Optional[str] = None
    lib_log_level: Optional[str] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            data = {}

        self.app_log_level = self._get_log_level(data, "app_log_level", "info")
        self.lib_log_level = self._get_log_level(data, "lib_log_level", "warning")

    def __eq__(self, other) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, LoggingConfig):
            return (
                self.app_log_level == other.app_log_level
                and self.lib_log_level == other.lib_log_level
            )
        return False

    def _get_log_level(self, data, key, default):
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

    def __init__(self, data: Optional[dict] = None):
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return

        self.product_docs_index_path = data.get("product_docs_index_path", None)
        self.product_docs_index_id = data.get("product_docs_index_id", None)

    def __eq__(self, other):
        """Compare two objects for equality."""
        if isinstance(other, ReferenceContent):
            return (
                self.product_docs_index_path == other.product_docs_index_path
                and self.product_docs_index_id == other.product_docs_index_id
            )
        return False

    def validate_yaml(self) -> None:
        """Validate reference content config."""
        if self.product_docs_index_path is not None:
            if os.path.exists(self.product_docs_index_path) is False:
                raise InvalidConfigurationError(
                    f"Reference content path '{self.product_docs_index_path}' does not exist"
                )
            if os.path.isfile(self.product_docs_index_path):
                raise InvalidConfigurationError(
                    f"Reference content path '{self.product_docs_index_path}' is not a directory"
                )
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


class OLSConfig(BaseModel):
    """OLS configuration."""

    conversation_cache: Optional[ConversationCacheConfig] = None
    logging_config: Optional[LoggingConfig] = None
    reference_content: Optional[ReferenceContent] = None

    default_provider: Optional[str] = None
    default_model: Optional[str] = None
    classifier_provider: Optional[str] = None
    classifier_model: Optional[str] = None
    summarizer_provider: Optional[str] = None
    summarizer_model: Optional[str] = None
    validator_provider: Optional[str] = None
    validator_model: Optional[str] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.conversation_cache = ConversationCacheConfig(
            data.get("conversation_cache", None)
        )
        self.default_provider = data.get("default_provider", None)
        self.default_model = data.get("default_model", None)
        self.classifier_provider = data.get(
            "classifier_provider", self.default_provider
        )
        self.classifier_model = data.get("classifier_model", self.default_model)
        self.summarizer_provider = data.get(
            "summarizer_provider", self.default_provider
        )
        self.summarizer_model = data.get("summarizer_model", self.default_model)
        self.validator_provider = data.get("validator_provider", self.default_provider)
        self.validator_model = data.get("validator_model", self.default_model)

        self.conversation_cache = ConversationCacheConfig(
            data.get("conversation_cache", None)
        )
        self.logging_config = LoggingConfig(data.get("logging_config", None))
        self.reference_content = ReferenceContent(data.get("reference_content", None))

    def __eq__(self, other) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, OLSConfig):
            return (
                self.conversation_cache == other.conversation_cache
                and self.logging_config == other.logging_config
                and self.default_provider == other.default_provider
                and self.default_model == other.default_model
                and self.classifier_provider == other.classifier_provider
                and self.classifier_model == other.classifier_model
                and self.summarizer_provider == other.summarizer_provider
                and self.summarizer_model == other.summarizer_model
                and self.validator_provider == other.validator_provider
                and self.validator_model == other.validator_model
                and self.reference_content == other.reference_content
            )
        return False

    def validate_yaml(self) -> None:
        """Validate OLS config."""
        if self.conversation_cache is None:
            raise InvalidConfigurationError("OSLConfig: conversation cache is not set")
        self.conversation_cache.validate_yaml()
        self.logging_config.validate_yaml()
        if self.reference_content is not None:
            self.reference_content.validate_yaml()


class DevConfig(BaseModel):
    """Developer-mode-only configuration options."""

    enable_dev_ui: bool = False
    disable_question_validation: bool = False
    llm_temperature_override: Optional[float] = None

    # TODO - wire this up once auth is implemented
    disable_auth: bool = False

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize developer configuration settings."""
        super().__init__()
        if data is None:
            return
        self.enable_dev_ui = str(data.get("enable_dev_ui", "False")).lower() == "true"
        self.disable_question_validation = (
            str(data.get("disable_question_validation", "False")).lower() == "true"
        )
        self.llm_temperature_override = data.get("llm_temperature_override", None)
        self.disable_auth = str(data.get("disable_auth", "False")).lower() == "true"

    def __eq__(self, other):
        """Compare two objects for equality."""
        if isinstance(other, DevConfig):
            return (
                self.enable_dev_ui == other.enable_dev_ui
                and self.disable_question_validation
                == other.disable_question_validation
                and self.llm_temperature_override == other.llm_temperature_override
                and self.disable_auth == other.disable_auth
            )
        return False

    def validate_yaml(self) -> None:
        """Validate OLS Dev config."""
        if self.llm_temperature_override is not None:
            if not isinstance(self.llm_temperature_override, (float, int)):
                raise InvalidConfigurationError(
                    "llm_temperature_override must be a float"
                )
            if self.llm_temperature_override < 0 or self.llm_temperature_override > 1:
                raise InvalidConfigurationError(
                    "llm_temperature_override must be between 0 and 1"
                )


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

    def __eq__(self, other) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, Config):
            return (
                self.ols_config == other.ols_config
                and self.llm_providers == other.llm_providers
            )
        return False

    def validate_yaml(self) -> None:
        """Validate all configurations."""
        if self.llm_providers is None:
            raise InvalidConfigurationError("no LLM providers config section found")
        self.llm_providers.validate_yaml()
        if self.ols_config is None:
            raise InvalidConfigurationError("no OLS config section found")
        self.ols_config.validate_yaml()
        self.dev_config.validate_yaml()
        for role in constants.PROVIDER_MODEL_ROLES:
            provider_attr_name = f"{role}_provider"
            provider_attr_value = getattr(self.ols_config, provider_attr_name, None)
            model_attr_name = f"{role}_model"
            model_attr_value = getattr(self.ols_config, model_attr_name, None)
            if isinstance(provider_attr_value, str) and isinstance(
                model_attr_value, str
            ):
                provider_config = self.llm_providers.providers.get(provider_attr_value)
                if provider_config is None:
                    raise InvalidConfigurationError(
                        f"{provider_attr_name} specifies an unknown provider {provider_attr_value}"
                    )
                model_config = provider_config.models.get(model_attr_value)
                if model_config is None:
                    raise InvalidConfigurationError(
                        f"{model_attr_name} specifies an unknown model {model_attr_value}"
                    )
            elif isinstance(provider_attr_value, str) and not isinstance(
                model_attr_value, str
            ):
                raise InvalidConfigurationError(
                    f"{provider_attr_name} is specified, but {model_attr_name} is missing"
                )
            elif not isinstance(provider_attr_value, str) and isinstance(
                model_attr_value, str
            ):
                raise InvalidConfigurationError(
                    f"{model_attr_name} is specified, but {provider_attr_name} is missing"
                )
