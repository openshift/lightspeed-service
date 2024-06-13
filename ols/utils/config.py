"""Configuration loader."""

import traceback
from io import TextIOBase
from typing import Any, Optional

import yaml

import ols.app.models.config as config_model
from ols.src.cache.cache import Cache
from ols.src.cache.cache_factory import CacheFactory

# as we the index_loader.py is excluded from type checks, it confuses
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
            cls._instance = super(AppConfig, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the class instance."""
        self.config = config_model.Config()
        self._query_filters: Optional[Redactor] = None
        self._rag_index: Optional[BaseIndex] = None
        self._conversation_cache: Optional[Cache] = None

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
        if self._conversation_cache is None:
            self._conversation_cache = CacheFactory.conversation_cache(
                self.ols_config.conversation_cache
            )
        return self._conversation_cache

    @property
    def query_redactor(self) -> Optional[Redactor]:
        """Return the query redactor."""
        # TODO: OLS-380 Config object mirrors configuration
        if self._query_filters is None:
            self._query_filters = Redactor(self.ols_config.query_filters)
        return self._query_filters

    @property
    def rag_index(self) -> Optional[BaseIndex]:
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
            raise


config: AppConfig = AppConfig()
