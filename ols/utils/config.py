"""Configuration loader."""

import traceback
from io import TextIOBase

import yaml
from llama_index.core.indices.base import BaseIndex

import ols.app.models.config as config_model
from ols.src.cache.cache import Cache
from ols.src.cache.cache_factory import CacheFactory
from ols.src.rag_index.index_loader import IndexLoader
from ols.utils.query_filter import QueryFilter


class ConfigManager:
    """Config manager class."""

    _instance = None  # Class-level variable to hold the singleton instance

    def __new__(cls):
        """Ensure only one instance of ConfigManager exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize instance attributes only if not already initialized."""
        if not hasattr(self, "config"):  # Check to avoid re-initialization
            self.config = None
        if not hasattr(self, "ols_config"):
            self.ols_config = None
        if not hasattr(self, "llm_config"):
            self.llm_config = None
        if not hasattr(self, "dev_config"):
            self.dev_config = None
        if not hasattr(self, "query_redactor"):
            self.query_redactor = None
        if not hasattr(self, "rag_index"):
            self.rag_index = None
        if not hasattr(self, "conversation_cache"):
            self.conversation_cache = None

    def init_empty_config(self) -> None:
        """Initialize empty configuration."""
        self.config = config_model.Config()
        self.ols_config = config_model.OLSConfig()
        self.llm_config = config_model.LLMProviders()
        self.dev_config = config_model.DevConfig()
        print(self.config)

    def init_config(self, config_file: str) -> None:
        """Load configuration from a YAML file."""
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                self.config = self.load_config_from_stream(f)
                self.ols_config = self.config.ols_config
                self.llm_config = self.config.llm_providers
                self.dev_config = self.config.dev_config

                self.conversation_cache = CacheFactory.conversation_cache(
                    self.ols_config.conversation_cache
                )
        except Exception as e:
            print(f"Failed to load config file {config_file}: {e!s}")
            print(traceback.format_exc())
            raise

    def load_config_from_stream(self, stream: TextIOBase) -> config_model.Config:
        """Load configuration from a YAML stream."""
        data = yaml.safe_load(stream)
        c = config_model.Config(data)
        c.validate_yaml()
        return c

    def init_query_filter(self) -> None:
        """Initialize question filter."""
        # TODO: OLS-380 Config object mirrors configuration
        self.query_redactor = QueryFilter()

    def init_vector_index(self) -> None:
        """Initialize vector index."""
        # TODO: OLS-380 Config object mirrors configuration
        self.rag_index = IndexLoader(self.ols_config.reference_content).vector_index

    def get_config(self) -> config_model.Config:
        """Get the current configuration."""
        if self.config is None:
            raise ValueError("Configuration has not been initialized.")
        return self.config

    def get_ols_config(self) -> config_model.OLSConfig:
        """Get the current ols config."""
        if self.config is None:
            raise ValueError("Configuration has not been initialized.")
        return self.ols_config

    def get_llm_config(self) -> config_model.LLMProviders:
        """Get the current llm config."""
        if self.config is None:
            raise ValueError("Configuration has not been initialized.")
        return self.llm_config

    def get_dev_config(self) -> config_model.DevConfig:
        """Get the current dev config."""
        if self.config is None:
            raise ValueError("Configuration has not been initialized.")
        return self.dev_config

    def get_conversation_cache(self) -> Cache:
        """Get the current conversation cache."""
        if self.config is None:
            raise ValueError("Configuration has not been initialized.")
        return self.conversation_cache

    def get_query_redactor(self) -> QueryFilter:
        """Get query filter."""
        if self.config is None:
            raise ValueError("Configuration has not been initialized.")
        return self.query_redactor

    def set_query_redactor(self, query_filter: QueryFilter) -> None:
        """Set query filter."""
        self.query_redactor = query_filter

    def get_rag_index(self) -> BaseIndex:
        """Get vector index."""
        if self.config is None:
            raise ValueError("Configuration has not been initialized.")
        return self.rag_index
