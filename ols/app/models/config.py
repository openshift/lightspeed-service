"""Config classes for the configuration structure."""

import logging
import os
import re
from typing import Any, Optional, Self

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    Field,
    FilePath,
    PositiveInt,
    PrivateAttr,
    field_validator,
    model_validator,
)

from ols import constants
from ols.utils import checks, tls


class ModelParameters(BaseModel):
    """Model parameters."""

    max_tokens_for_response: PositiveInt = constants.DEFAULT_MAX_TOKENS_FOR_RESPONSE
    max_tokens_per_tool_output: PositiveInt = (
        constants.DEFAULT_MAX_TOKENS_PER_TOOL_OUTPUT
    )
    max_tokens_for_tools: PositiveInt = constants.DEFAULT_MAX_TOKENS_FOR_TOOLS


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
            raise checks.InvalidConfigurationError("model name is missing")

        data["credentials"] = checks.read_secret(
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
            raise checks.InvalidConfigurationError("model options must be dictionary")
        for key in options:
            if not isinstance(key, str):
                raise checks.InvalidConfigurationError(
                    "key for model option must be string"
                )
        return options

    @model_validator(mode="after")
    def validate_context_window_and_max_tokens(self) -> Self:
        """Validate context window size and max tokens for response."""
        if self.context_window_size <= self.parameters.max_tokens_for_response:  # type: ignore [operator]
            raise checks.InvalidConfigurationError(
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
            self.tls_key_password = checks.get_attribute_from_file(
                data, "tls_key_password_path"
            )

    def validate_yaml(self, disable_tls: bool = False) -> None:
        """Validate TLS config."""
        if not disable_tls and not self._ignore_missing_certs:
            if not self.tls_certificate_path:
                raise checks.InvalidConfigurationError(
                    "Can not enable TLS without ols_config.tls_config.tls_certificate_path"
                )

            checks.file_check(self.tls_certificate_path, "OLS server certificate")
            if not self.tls_key_path:
                raise checks.InvalidConfigurationError(
                    "Can not enable TLS without ols_config.tls_config.tls_key_path"
                )
            checks.file_check(self.tls_key_path, "OLS server certificate private key")


class ProxyConfig(BaseModel):
    """HTTPS Proxy configuration."""

    proxy_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
    )
    proxy_ca_cert_path: Optional[FilePath] = None
    no_proxy_hosts: Optional[list[str]] = Field(
        default_factory=lambda: [
            host for host in os.getenv("no_proxy", "").split(",") if host
        ]
    )

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if not data:
            return
        if "proxy_url" in data:
            # avoid overwriting the proxy_url set by environment variable
            self.proxy_url = data.get("proxy_url")
        self.proxy_ca_cert_path = data.get("proxy_ca_cert_path")
        if "no_proxy_hosts" in data:
            # avoid overwriting the no_proxy_hosts set by environment variable
            self.no_proxy_hosts = data.get("no_proxy_hosts", self.no_proxy_hosts)

    def validate_yaml(self) -> None:
        """Validate proxy config."""
        if self.proxy_url is None and self.proxy_ca_cert_path:
            raise checks.InvalidConfigurationError("Proxy URL is missing")
        if self.proxy_url and not checks.is_valid_http_url(self.proxy_url):
            raise checks.InvalidConfigurationError(
                "Proxy URL is invalid, only http:// and https:// URLs are supported"
            )
        if self.proxy_ca_cert_path is not None:
            checks.file_check(self.proxy_ca_cert_path, "Proxy CA certificate")

    def is_https(self) -> bool:
        """Check if the proxy URL is HTTPS."""
        if self.proxy_url is None:
            return False
        return self.proxy_url.lower().startswith("https://")

    @model_validator(mode="before")
    @classmethod
    def set_default_proxy_url(cls, values: Any) -> Any:
        """Set proxy URL from environment variable if not provided."""
        if "proxy_url" not in values or values["proxy_url"] is None:
            env_proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
            if env_proxy:
                values["proxy_url"] = env_proxy
        return values


class AuthenticationConfig(BaseModel):
    """Authentication configuration."""

    module: Optional[str] = None
    skip_tls_verification: bool = False
    k8s_cluster_api: Optional[AnyHttpUrl] = None
    k8s_ca_cert_path: Optional[FilePath] = None

    def validate_yaml(self) -> None:
        """Validate YAML containing authentication configuration section."""
        if self.module is None:
            raise checks.InvalidConfigurationError("Authentication module is not setup")
        if self.module not in constants.SUPPORTED_AUTHENTICATION_MODULES:
            raise checks.InvalidConfigurationError(
                f"invalid authentication module: {self.module}, supported modules are"
                f" {constants.SUPPORTED_AUTHENTICATION_MODULES}"
            )


class TLSSecurityProfile(BaseModel):
    """TLS security profile structure."""

    profile_type: Optional[str] = None
    min_tls_version: Optional[str] = None
    ciphers: Optional[list[str]] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is not None:
            self.profile_type = data.get("type")
            self.min_tls_version = data.get("minTLSVersion")
            self.ciphers = data.get("ciphers")

    def validate_yaml(self) -> None:
        """Validate structure content."""
        # check the TLS profile type
        if self.profile_type is not None:
            try:
                tls.TLSProfiles(self.profile_type)
            except ValueError:
                raise checks.InvalidConfigurationError(
                    f"Invalid TLS profile type '{self.profile_type}'"
                )
        # check the TLS protocol version
        if self.min_tls_version is not None:
            try:
                tls.TLSProtocolVersion(self.min_tls_version)
            except ValueError:
                raise checks.InvalidConfigurationError(
                    f"Invalid minimal TLS version '{self.min_tls_version}'"
                )
        # check ciphers
        if self.ciphers is not None:
            # just perform the check for non-custom TLS profile type
            if self.profile_type is not None and self.profile_type != "Custom":
                supported_ciphers = tls.TLS_CIPHERS[tls.TLSProfiles(self.profile_type)]
                for cipher in self.ciphers:
                    if cipher not in supported_ciphers:
                        raise checks.InvalidConfigurationError(
                            f"Unsupported cipher '{cipher}' found in configuration"
                        )


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


class FakeConfig(ProviderSpecificConfig, extra="forbid"):
    """Configuration specific to fake provider."""

    stream: Optional[bool]
    response: Optional[str]
    chunks: Optional[int]
    sleep: Optional[float]


class ProviderConfig(BaseModel):
    """LLM provider configuration."""

    name: Optional[str] = None
    type: Optional[str] = None
    url: Optional[AnyHttpUrl] = None
    credentials: Optional[str] = None
    project_id: Optional[str] = None
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None
    openai_config: Optional[OpenAIConfig] = None
    azure_config: Optional[AzureOpenAIConfig] = None
    watsonx_config: Optional[WatsonxConfig] = None
    bam_config: Optional[BAMConfig] = None
    rhoai_vllm_config: Optional[RHOAIVLLMConfig] = None
    rhelai_vllm_config: Optional[RHELAIVLLMConfig] = None
    fake_provider_config: Optional[FakeConfig] = None
    certificates_store: Optional[str] = None
    tls_security_profile: Optional[TLSSecurityProfile] = None

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
            self.credentials = checks.read_secret(
                data, constants.CREDENTIALS_PATH_SELECTOR, constants.API_TOKEN_FILENAME
            )
        except FileNotFoundError:
            if ignore_llm_secrets:
                self.credentials = None
            else:
                raise

        # OLS-622: Provider-specific configuration parameters in configuration file
        self.project_id = data.get("project_id", None)
        if self.type == constants.PROVIDER_WATSONX and self.project_id is None:
            raise checks.InvalidConfigurationError(
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
        if self.type in (
            constants.PROVIDER_RHOAI_VLLM,
            constants.PROVIDER_RHELAI_VLLM,
            constants.PROVIDER_OPENAI,
            constants.PROVIDER_AZURE_OPENAI,
        ):
            self.certificates_store = os.path.join(
                certificate_directory, constants.CERTIFICATE_STORAGE_FILENAME
            )
        self.tls_security_profile = TLSSecurityProfile(
            data.get("tlsSecurityProfile", None)
        )

    def set_provider_type(self, data: dict) -> None:
        """Set the provider type."""
        # Default provider type to be the provider name, unless
        # specified explicitly.
        self.type = str(data.get("type", self.name)).lower()
        if self.type not in constants.SUPPORTED_PROVIDER_TYPES:
            raise checks.InvalidConfigurationError(
                f"invalid provider type: {self.type}, supported types are"
                f" {set(constants.SUPPORTED_PROVIDER_TYPES)}"
            )

    def setup_models_config(self, data: dict) -> None:
        """Set up models configuration."""
        if "models" not in data or len(data["models"]) == 0:
            raise checks.InvalidConfigurationError(
                f"no models configured for provider {data['name']}"
            )
        for m in data["models"]:
            if "name" not in m:
                raise checks.InvalidConfigurationError("model name is missing")
            # add provider to model data - needed for some constants
            # resolution based on the specific provider
            m["provider"] = self.type
            model = ModelConfig(**m)
            self.models[m["name"]] = model

    def set_provider_specific_configuration(  # noqa: C901  # pylint: disable=R0912
        self, data: dict
    ) -> None:
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
            raise checks.InvalidConfigurationError(
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
                        azure_config["tenant_id"] = checks.read_secret(
                            azure_config,
                            constants.CREDENTIALS_PATH_SELECTOR,
                            constants.AZURE_TENANT_ID_FILENAME,
                            directory_name_expected=True,
                            raise_on_error=False,
                        )
                        azure_config["client_id"] = checks.read_secret(
                            azure_config,
                            constants.CREDENTIALS_PATH_SELECTOR,
                            constants.AZURE_CLIENT_ID_FILENAME,
                            directory_name_expected=True,
                            raise_on_error=False,
                        )
                        azure_config["client_secret"] = checks.read_secret(
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
                case constants.PROVIDER_FAKE:
                    fake_provider_config = data.get("fake_provider_config")
                    self.fake_provider_config = FakeConfig(**fake_provider_config)
                case _:
                    raise checks.InvalidConfigurationError(
                        f"Unsupported provider {self.type} configured"
                    )

    @staticmethod
    def read_api_key(config: Optional[dict]) -> None:
        """Read API key from file with secret."""
        if config is None:
            return
        config["api_key"] = checks.read_secret(
            config,
            constants.CREDENTIALS_PATH_SELECTOR,
            constants.API_TOKEN_FILENAME,
            raise_on_error=False,
        )

    def check_provider_config(self, provider_config: Any) -> None:
        """Check if configuration is presented for selected provider type."""
        if provider_config is None:
            raise checks.InvalidConfigurationError(
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
                and self.tls_security_profile == other.tls_security_profile
            )
        return False

    def validate_yaml(self) -> None:
        """Validate provider config."""
        if self.name is None:
            raise checks.InvalidConfigurationError("provider name is missing")
        if self.url is not None and not checks.is_valid_http_url(self.url):
            raise checks.InvalidConfigurationError(
                "provider URL is invalid, only http:// and https:// URLs are supported"
            )


class LLMProviders(BaseModel):
    """LLM providers configuration."""

    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

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
                raise checks.InvalidConfigurationError("provider name is missing")
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


class MCPServerConfig(BaseModel):
    """MCP server configuration.

    MCP (Model Context Protocol) servers provide tools and capabilities to the
    AI agents. These are configured by this structure. Only MCP servers
    defined in the olsconfig.yaml configuration are available to the agents.
    """

    name: str = Field(
        title="MCP name",
        description="MCP server name that must be unique",
    )

    url: str = Field(
        title="MCP server URL",
        description="URL of the MCP server",
    )

    timeout: Optional[int] = Field(
        default=None,
        title="Request timeout",
        description=(
            "Timeout in seconds for requests to the MCP server. "
            "If not specified, the default timeout will be used."
        ),
    )

    headers: dict[str, str] = Field(
        default_factory=dict,
        title="Authorization headers",
        description=(
            "Headers to send to the MCP server. "
            "The map contains the header name and the path to a file containing "
            "the header value (secret). "
            "There are 2 special cases: "
            f"1. Usage of the kubernetes token in the header. "
            f"To specify this use a string '{constants.MCP_KUBERNETES_PLACEHOLDER}' "
            f"instead of the file path. "
            f"2. Usage of the client provided token in the header. "
            f"To specify this use a string '{constants.MCP_CLIENT_PLACEHOLDER}' "
            f"instead of the file path."
        ),
    )

    _resolved_headers: dict[str, str] = PrivateAttr(default_factory=dict)

    @property
    def resolved_headers(self) -> dict[str, str]:
        """Resolved headers (computed from headers)."""
        return self._resolved_headers


class ToolFilteringConfig(BaseModel):
    """Configuration for tool filtering using hybrid RAG retrieval.

    If this config is present, tool filtering is enabled. If absent, all tools are used.
    """

    embed_model_path: Optional[str] = Field(
        default=None,
        description="Path to sentence transformer model for embeddings",
    )

    alpha: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Weight for dense vs sparse retrieval (1.0 = full dense, 0.0 = full sparse)",
    )

    top_k: int = Field(
        default=10, ge=1, le=50, description="Number of tools to retrieve"
    )

    threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for filtering results",
    )


class MCPServers(BaseModel):
    """MCP servers configuration."""

    servers: list[MCPServerConfig] = []

    @model_validator(mode="after")
    def check_duplicite_servers(self) -> Self:
        """Check if there are duplicate servers."""
        server_names = set()
        for server in self.servers:
            if server.name in server_names:
                raise ValueError(f"Duplicate server name: '{server.name}'")
            server_names.add(server.name)
        return self


class PostgresConfig(BaseModel):
    """Postgres configuration."""

    host: str = constants.POSTGRES_CACHE_HOST
    port: PositiveInt = constants.POSTGRES_CACHE_PORT
    dbname: str = constants.POSTGRES_CACHE_DBNAME
    user: str = constants.POSTGRES_CACHE_USER
    password_path: Optional[FilePath] = None
    password: Optional[str] = None
    ssl_mode: str = constants.POSTGRES_CACHE_SSL_MODE
    gss_encmode: str = constants.POSTGRES_CACHE_GSSENCMODE
    ca_cert_path: Optional[FilePath] = None
    max_entries: PositiveInt = constants.POSTGRES_CACHE_MAX_ENTRIES

    def __init__(self, **data: Any) -> None:
        """Initialize configuration."""
        super().__init__(**data)
        # password should be read from file
        if self.password_path is not None:
            with open(self.password_path, "r", encoding="utf-8") as f:
                self.password = f.read().rstrip()

    @model_validator(mode="after")
    def validate_yaml(self) -> Self:
        """Validate Postgres cache config."""
        if not 0 < self.port < 65536:
            raise ValueError("The port needs to be between 0 and 65536")
        return self


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
            raise checks.InvalidConfigurationError(
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
            raise checks.InvalidConfigurationError(
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
            raise checks.InvalidConfigurationError("name is missing")
        if self.pattern is None:
            raise checks.InvalidConfigurationError("pattern is missing")
        try:
            re.compile(self.pattern)
        except re.error as e:
            raise checks.InvalidConfigurationError("pattern is invalid") from e
        if self.replace_with is None:
            raise checks.InvalidConfigurationError("replace_with is missing")


class ConversationCacheConfig(BaseModel):
    """Conversation cache configuration."""

    type: Optional[str] = None
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
                case constants.CACHE_TYPE_MEMORY:
                    if constants.CACHE_TYPE_MEMORY not in data:
                        raise checks.InvalidConfigurationError(
                            "memory conversation cache type is specified,"
                            " but memory configuration is missing"
                        )
                    self.memory = InMemoryCacheConfig(
                        data.get(constants.CACHE_TYPE_MEMORY)
                    )
                case constants.CACHE_TYPE_POSTGRES:
                    if constants.CACHE_TYPE_POSTGRES not in data:
                        raise checks.InvalidConfigurationError(
                            "Postgres conversation cache type is specified,"
                            " but Postgres configuration is missing"
                        )
                    self.postgres = PostgresConfig(
                        **data.get(constants.CACHE_TYPE_POSTGRES)
                    )
                case _:
                    raise checks.InvalidConfigurationError(
                        f"unknown conversation cache type: {self.type}"
                    )

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, ConversationCacheConfig):
            return (
                self.type == other.type
                and self.memory == other.memory
                and self.postgres == other.postgres
            )
        return False

    def validate_yaml(self) -> None:
        """Validate conversation cache config."""
        if self.type is None:
            raise checks.InvalidConfigurationError("missing conversation cache type")
        # cache type is specified, we can decide which cache configuration to validate
        match self.type:
            case constants.CACHE_TYPE_MEMORY:
                self.memory.validate_yaml()
            case constants.CACHE_TYPE_POSTGRES:
                pass  # it is validated by Pydantic already
            case _:
                raise checks.InvalidConfigurationError(
                    f"unknown conversation cache type: {self.type}"
                )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    app_log_level: int = logging.INFO
    lib_log_level: int = logging.WARNING
    uvicorn_log_level: int = logging.WARNING
    suppress_metrics_in_log: bool = False
    suppress_auth_checks_warning_in_log: bool = False

    def __init__(self, **data: Any) -> None:
        """Initialize configuration and perform basic validation."""
        # convert input strings (level names, eg. debug/info,...) to
        # logging level names (integer values) for defined model fields
        for field in filter(lambda x: x.endswith("_log_level"), self.model_fields):
            if field in data:
                data[field] = checks.get_log_level(data[field])
        super().__init__(**data)


class ReferenceContentIndex(BaseModel):
    """Reference content index configuration."""

    product_docs_index_path: Optional[FilePath] = None
    product_docs_index_id: Optional[str] = None
    product_docs_origin: Optional[str] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.product_docs_index_path = data.get("product_docs_index_path", None)
        self.product_docs_index_id = data.get("product_docs_index_id", None)
        self.product_docs_origin = data.get("product_docs_origin", None)

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, ReferenceContentIndex):
            return (
                self.product_docs_index_path == other.product_docs_index_path
                and self.product_docs_index_id == other.product_docs_index_id
                and self.product_docs_origin == other.product_docs_origin
            )
        return False

    def validate_yaml(self) -> None:
        """Validate reference content index config."""
        if self.product_docs_index_path is not None:
            try:
                checks.dir_check(
                    self.product_docs_index_path, "Reference content index path"
                )
            except checks.InvalidConfigurationError as e_original:
                # special case for OCP documents not having the current version.
                # we use the "latest" version from the vector store directory.
                default_path = os.path.join(
                    os.path.dirname(self.product_docs_index_path), "latest"
                )
                try:
                    checks.dir_check(default_path, "Reference content index path")
                    self.product_docs_index_path = FilePath(default_path)
                    # we don't know the index_id for "latest" so ignore what
                    # the config says and load whatever index is there
                    self.product_docs_index_id = None
                except checks.InvalidConfigurationError:
                    raise e_original
        else:
            if self.product_docs_index_id is not None:
                raise checks.InvalidConfigurationError(
                    "product_docs_index_id is specified but product_docs_index_path is missing"
                )


class ReferenceContent(BaseModel):
    """Reference content configuration."""

    embeddings_model_path: Optional[FilePath] = None
    indexes: Optional[list[ReferenceContentIndex]] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return

        self.embeddings_model_path = data.get("embeddings_model_path", None)
        if "indexes" in data:
            self.indexes = [ReferenceContentIndex(i) for i in data["indexes"]]
        else:
            self.indexes = None

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, ReferenceContent):
            return (
                self.indexes == other.indexes
                and self.embeddings_model_path == other.embeddings_model_path
            )

        return False

    def validate_yaml(self) -> None:
        """Validate reference content config."""
        if self.embeddings_model_path is not None:
            checks.dir_check(self.embeddings_model_path, "Embeddings model path")
        if self.indexes is not None:
            for index in self.indexes:
                index.validate_yaml()


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

    @property
    def config_status_enabled(self) -> bool:
        """Config status is enabled when feedback or transcripts collection is enabled."""
        return not self.feedback_disabled or not self.transcripts_disabled

    @property
    def config_status_storage(self) -> Optional[str]:
        """Infer config status storage from feedback or transcripts storage."""
        if not self.config_status_enabled:
            return None
        base_storage = self.feedback_storage or self.transcripts_storage
        if base_storage is None:
            return None
        return os.path.join(os.path.dirname(base_storage), "config_status")


class SchedulerConfig(BaseModel):
    """Scheduler configuration."""

    period: int


class LimiterConfig(BaseModel):
    """Configuration for one quota limiter."""

    type: Optional[str]
    initial_quota: Optional[int]
    quota_increase: Optional[int]
    period: Optional[str]


class LimitersConfig(BaseModel):
    """Configuration for all quota limiters."""

    limiters: dict[str, LimiterConfig] = {}

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        # convert list of limiters into a dictionary
        for limiter in data:
            if "name" not in limiter:
                raise checks.InvalidConfigurationError("limiter name is missing")
            limiter_config = LimiterConfig(**limiter)
            self.limiters[limiter["name"]] = limiter_config


class QuotaHandlersConfig(BaseModel):
    """Quota limiter configuration."""

    storage: Optional[PostgresConfig] = None
    scheduler: Optional[SchedulerConfig] = None
    limiters: Optional[LimitersConfig] = None
    enable_token_history: Optional[bool] = None

    def __init__(self, data: Optional[dict] = None) -> None:
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        # check the content
        if "storage" not in data:
            raise checks.InvalidConfigurationError(
                "Missing storage configuration for quota limiters"
            )
        if "scheduler" not in data:
            raise checks.InvalidConfigurationError(
                "Missing scheduler configuration for quota limiters"
            )
        # setup all subcategories
        self.storage = PostgresConfig(**data.get("storage"))
        self.scheduler = SchedulerConfig(**data.get("scheduler"))
        self.limiters = LimitersConfig(data.get("limiters", None))
        self.enable_token_history = data.get("enable_token_history", False)


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
    expire_llm_is_ready_persistent_state: Optional[int] = -1
    max_workers: Optional[int] = None
    query_filters: Optional[list[QueryFilter]] = None
    query_validation_method: Optional[str] = constants.QueryValidationMethod.DISABLED

    user_data_collection: UserDataCollection = UserDataCollection()
    tls_security_profile: Optional[TLSSecurityProfile] = None

    extra_ca: list[FilePath] = []
    certificate_directory: Optional[str] = None

    quota_handlers: Optional[QuotaHandlersConfig] = None

    proxy_config: Optional[ProxyConfig] = None

    tool_filtering: Optional[ToolFilteringConfig] = None

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
        self.max_workers = data.get("max_workers", 1)
        self.expire_llm_is_ready_persistent_state = data.get(
            "expire_llm_is_ready_persistent_state", -1
        )
        self.authentication_config = AuthenticationConfig(
            **data.get("authentication_config", {})
        )
        # setup the authentication module, which is optional in configuration file
        if self.authentication_config.module is None:
            self.authentication_config.module = constants.DEFAULT_AUTHENTICATION_MODULE

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
        self.system_prompt = checks.get_attribute_from_file(data, "system_prompt_path")

        self.extra_ca = data.get("extra_ca", [])
        self.certificate_directory = data.get(
            "certificate_directory", constants.DEFAULT_CERTIFICATE_DIRECTORY
        )
        self.tls_security_profile = TLSSecurityProfile(
            data.get("tlsSecurityProfile", None)
        )
        self.quota_handlers = QuotaHandlersConfig(data.get("quota_handlers", None))
        self.proxy_config = ProxyConfig(data.get("proxy_config"))
        if data.get("tool_filtering", None) is not None:
            self.tool_filtering = ToolFilteringConfig(**data.get("tool_filtering"))

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, OLSConfig):
            return (
                self.conversation_cache == other.conversation_cache
                and self.logging_config == other.logging_config
                and self.reference_content == other.reference_content
                and self.default_provider == other.default_provider
                and self.default_model == other.default_model
                and self.max_workers == other.max_workers
                and self.query_filters == other.query_filters
                and self.query_validation_method == other.query_validation_method
                and self.tls_config == other.tls_config
                and self.certificate_directory == other.certificate_directory
                and self.system_prompt == other.system_prompt
                and self.system_prompt_path == other.system_prompt_path
                and self.tls_security_profile == other.tls_security_profile
                and self.authentication_config == other.authentication_config
                and self.expire_llm_is_ready_persistent_state
                == other.expire_llm_is_ready_persistent_state
                and self.quota_handlers == other.quota_handlers
                and self.proxy_config == other.proxy_config
                and self.tool_filtering == other.tool_filtering
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
        if self.tls_security_profile is not None:
            self.tls_security_profile.validate_yaml()
        if self.authentication_config is not None:
            self.authentication_config.validate_yaml()
        if self.proxy_config is not None:
            self.proxy_config.validate_yaml()

        valid_query_validation_methods = list(constants.QueryValidationMethod)
        if self.query_validation_method not in valid_query_validation_methods:
            raise checks.InvalidConfigurationError(
                f"Invalid query validation method: {self.query_validation_method}\n"
                f"Available options are {valid_query_validation_methods}"
            )


class DevConfig(BaseModel):
    """Developer-mode-only configuration options."""

    enable_dev_ui: bool = False
    llm_params: dict = {}
    disable_auth: bool = False
    disable_tls: bool = False
    pyroscope_url: Optional[str] = None
    k8s_auth_token: Optional[str] = None
    run_on_localhost: bool = False
    enable_system_prompt_override: bool = False
    uvicorn_port_number: Optional[int] = None

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, DevConfig):
            return (
                self.enable_dev_ui == other.enable_dev_ui
                and self.llm_params == other.llm_params
                and self.disable_auth == other.disable_auth
                and self.disable_tls == other.disable_tls
                and self.pyroscope_url == other.pyroscope_url
                and self.k8s_auth_token == other.k8s_auth_token
                and self.run_on_localhost == other.run_on_localhost
                and self.enable_system_prompt_override
                == other.enable_system_prompt_override
            )
        return False


class Config(BaseModel):
    """Global service configuration."""

    llm_providers: LLMProviders = LLMProviders()
    ols_config: OLSConfig = OLSConfig()
    dev_config: DevConfig = DevConfig()
    mcp_servers: MCPServers = MCPServers()

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
            raise checks.InvalidConfigurationError("no OLS config section found")
        v = data.get("llm_providers")
        if v is not None:
            self.llm_providers = LLMProviders(
                v, ignore_llm_secrets, self.ols_config.certificate_directory
            )
        else:
            raise checks.InvalidConfigurationError(
                "no LLM providers config section found"
            )

        # initialize MCP servers
        self.mcp_servers = MCPServers(servers=data.get("mcp_servers", []))

        # Validate MCP servers now that auth config is available
        self._validate_mcp_servers()

        # Always initialize dev config, even if there's no config for it.
        self.dev_config = DevConfig(**data.get("dev_config", {}))

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, Config):
            return (
                self.ols_config == other.ols_config
                and self.llm_providers == other.llm_providers
                and self.mcp_servers == other.mcp_servers
            )
        return False

    def _validate_default_provider_and_model(self) -> None:
        selected_default_provider = self.ols_config.default_provider
        selected_default_model = self.ols_config.default_model

        provider_specified = selected_default_provider is not None
        model_specified = selected_default_model is not None

        if not provider_specified:
            raise checks.InvalidConfigurationError("default_provider is missing")
        if not model_specified:
            raise checks.InvalidConfigurationError("default_model is missing")

        # provider and model are specified
        provider_config = self.llm_providers.providers.get(selected_default_provider)
        if provider_config is None:
            raise checks.InvalidConfigurationError(
                f"default_provider specifies an unknown provider {selected_default_provider}"
            )
        model_config = provider_config.models.get(selected_default_model)
        if model_config is None:
            raise checks.InvalidConfigurationError(
                f"default_model specifies an unknown model {selected_default_model}"
            )

    def _validate_mcp_servers(self) -> None:
        """Validate MCP servers with auth module context.

        Filters out servers where authorization headers cannot be resolved.
        """
        auth_module = getattr(
            getattr(self.ols_config, "authentication_config", None),
            "module",
            None,
        )
        self.mcp_servers.servers = checks.validate_mcp_servers(
            self.mcp_servers.servers,
            auth_module,
        )

    def validate_yaml(self) -> None:
        """Validate all configurations."""
        self.llm_providers.validate_yaml()
        self.ols_config.validate_yaml(self.dev_config.disable_tls)
        self._validate_default_provider_and_model()
