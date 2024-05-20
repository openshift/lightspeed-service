"""Configuration loader."""

import traceback
from io import TextIOBase
from typing import Any, Optional

import yaml
from llama_index.core.indices.base import BaseIndex

import ols.app.models.config as config_model
from ols.src.cache.cache import Cache
from ols.src.cache.cache_factory import CacheFactory
from ols.src.rag_index.index_loader import IndexLoader
from ols.utils.query_filter import QueryFilters


class AppConfig:
    """Singleton class to load and store the configuration."""

    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "AppConfig":
        """Create a new instance of the class."""
        if not isinstance(cls._instance, cls):
            cls._instance = super(AppConfig, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the class instance."""
        self.config = config_model.Config()
        self._query_filters: Optional[QueryFilters] = None
        self._rag_index: Optional[BaseIndex] = None

    @property
    def llm_config(self) -> config_model.LLMProviders:
        """Return the LLM providers configuration."""
        return self.config.llm_providers

    @property
    def ols_config(self) -> config_model.OLSConfig:
        """Return the OLS configuration."""
        return self.config.ols_config

    @property
    def dev_config(self) -> config_model.DevConfig:
        """Return the dev configuration."""
        return self.config.dev_config

    @property
    def conversation_cache(self) -> Cache:
        """Return the conversation cache."""
        if self.ols_config.conversation_cache is None:
            raise ValueError("Conversation cache configuration is not set in config")
        return CacheFactory.conversation_cache(self.ols_config.conversation_cache)

    @property
    def query_redactor(self) -> Optional[QueryFilters]:
        """Return the query redactor."""
        # TODO: OLS-380 Config object mirrors configuration
        if self._query_filters is None:
            self._query_filters = QueryFilters(self.ols_config.query_filters)
        return self._query_filters

    @property
    def rag_index(self) -> Optional[BaseIndex[Any]]:
        """Return the RAG index."""
        # TODO: OLS-380 Config object mirrors configuration
        if self._rag_index is None:
            self._rag_index = IndexLoader(
                self.ols_config.reference_content
            ).vector_index
        return self._rag_index

    def reload_empty(self) -> None:
        """Reload the configuration with empty values."""
        self.config = config_model.Config()

    @staticmethod
    def _load_config_from_yaml_stream(stream: TextIOBase) -> config_model.Config:
        """Load configuration from a YAML stream."""
        data = yaml.safe_load(stream)
        config = config_model.Config(data)
        config.validate_yaml()
        return config

    def reload_from_yaml_file(self, config_file: str) -> None:
        """Reload the configuration from the YAML file."""
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                self.config = self._load_config_from_yaml_stream(f)
            # reset the query filters and rag index to not use cached
            # values
            self._query_filters = None
            self._rag_index = None
        except Exception as e:
            print(f"Failed to load config file {config_file}: {e!s}")
            print(traceback.format_exc())
            raise e


config: AppConfig = AppConfig()
