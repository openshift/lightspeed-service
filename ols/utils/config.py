"""Configuration loader."""

import traceback
from io import TextIOBase
from typing import Any, Optional

import yaml

import ols.app.models.config as config_model
from ols.src.cache.cache import Cache
from ols.src.cache.cache_factory import CacheFactory
from ols.src.quota.quota_limiter import QuotaLimiter
from ols.src.quota.quota_limiter_factory import QuotaLimiterFactory
from ols.src.quota.token_usage_history import TokenUsageHistory

# as the index_loader.py is excluded from type checks, it confuses
# mypy a bit, hence the [attr-defined] bellow
from ols.src.rag_index.index_loader import IndexLoader  # type: ignore [attr-defined]
from ols.utils.redactor import Redactor

# NOTE: Loading/importing something from llama_index bumps memory
# consumption up to ~400MiB.
# from llama_index.core.indices.base import BaseIndex
# Here, we need it just for typing, so we use Any instead.
BaseIndex = Any


class AppConfig:
    """Singleton class to load and store the configuration."""

    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "AppConfig":
        """Create a new instance of the class."""
        if not isinstance(cls._instance, cls):
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the class instance."""
        self.config = config_model.Config()
        self._query_filters: Optional[Redactor] = None
        self._rag_index_loader: Optional[IndexLoader] = None
        self._conversation_cache: Optional[Cache] = None
        self._quota_limiters: Optional[list[QuotaLimiter]] = None
        self._token_usage_history: Optional[TokenUsageHistory] = None

    @property
    def llm_config(self) -> config_model.LLMProviders:
        """Return the LLM providers configuration."""
        return self.config.llm_providers

    @property
    def ols_config(self) -> config_model.OLSConfig:
        """Return the OLS configuration."""
        return self.config.ols_config

    @property
    def mcp_servers(self) -> config_model.MCPServers:
        """Return the MCP servers configuration."""
        return self.config.mcp_servers

    @property
    def dev_config(self) -> config_model.DevConfig:
        """Return the dev configuration."""
        return self.config.dev_config

    @property
    def user_data_collector_config(self) -> config_model.UserDataCollectorConfig:
        """Return the user data collector configuration."""
        return self.config.user_data_collector_config

    @property
    def conversation_cache(self) -> Cache:
        """Return the conversation cache."""
        if self._conversation_cache is None:
            self._conversation_cache = CacheFactory.conversation_cache(
                self.ols_config.conversation_cache
            )
        return self._conversation_cache

    @property
    def quota_limiters(self) -> list[QuotaLimiter]:
        """Return all quota limiters."""
        if self._quota_limiters is None:
            self._quota_limiters = QuotaLimiterFactory.quota_limiters(
                self.ols_config.quota_handlers
            )
        return self._quota_limiters

    @property
    def token_usage_history(self) -> Optional[TokenUsageHistory]:
        """Return token usage history object."""
        if (
            self._token_usage_history is None
            and self.ols_config.quota_handlers is not None
            and self.ols_config.quota_handlers.enable_token_history
        ):
            self._token_usage_history = TokenUsageHistory(
                self.ols_config.quota_handlers.storage
            )
        return self._token_usage_history

    @property
    def query_redactor(self) -> Redactor:
        """Return the query redactor."""
        # TODO: OLS-380 Config object mirrors configuration
        if self._query_filters is None:
            self._query_filters = Redactor(self.ols_config.query_filters)
        return self._query_filters

    @property
    def rag_index(self) -> Optional[BaseIndex]:
        """Return the RAG index."""
        # TODO: OLS-380 Config object mirrors configuration
        return self.rag_index_loader.vector_indexes

    @property
    def rag_index_loader(self) -> Optional[IndexLoader]:
        """Return the RAG index loader."""
        if self._rag_index_loader is None:
            self._rag_index_loader = IndexLoader(self.ols_config.reference_content)
        return self._rag_index_loader

    def reload_empty(self) -> None:
        """Reload the configuration with empty values."""
        self.config = config_model.Config()

    @staticmethod
    def _load_config_from_yaml_stream(
        stream: TextIOBase,
        ignore_llm_secrets: bool = False,
        ignore_missing_certs: bool = False,
    ) -> config_model.Config:
        """Load configuration from a YAML stream."""
        data = yaml.safe_load(stream)
        config = config_model.Config(data, ignore_llm_secrets, ignore_missing_certs)
        config.validate_yaml()
        return config

    def reload_from_yaml_file(
        self,
        config_file: str,
        ignore_llm_secrets: bool = False,
        ignore_missing_certs: bool = False,
    ) -> None:
        """Reload the configuration from the YAML file."""
        try:
            with open(config_file, encoding="utf-8") as f:
                self.config = self._load_config_from_yaml_stream(
                    f, ignore_llm_secrets, ignore_missing_certs
                )
            # reset the query filters and rag index to not use cached
            # values
            self._query_filters = None
            self._rag_index_loader = None
        except Exception as e:
            print(f"Failed to load config file {config_file}: {e!s}")
            print(traceback.format_exc())
            raise


config: AppConfig = AppConfig()
