"""Configuration loader."""

import traceback
from io import TextIOBase

import yaml

import ols.app.models.config as config_model
from ols.src.cache.cache_factory import CacheFactory
from ols.src.rag_index.index_loader import IndexLoader
from ols.utils.query_filter import QueryFilter

config = None
ols_config = None
llm_config = None
dev_config = None
conversation_cache = None
query_redactor = None
rag_index = None


def init_empty_config() -> None:
    """Initialize empty configuration."""
    # TODO: understand use case
    global config
    global ols_config
    global llm_config
    global dev_config


def load_config_from_stream(stream: TextIOBase) -> config_model.Config:
    """Load configuration from a YAML stream."""
    ## TODO: duplicates to be handled
    data = yaml.safe_load(stream)
    # print(data)
    c = config_model.Config(**data)
    # c.validate_yaml()
    return c


def init_config(config_file: str) -> None:
    """Load configuration from a YAML file."""
    global config
    global ols_config
    global llm_config
    global dev_config
    global conversation_cache

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = load_config_from_stream(f)
            ols_config = config.ols_config
            llm_config = config.llm_providers
            dev_config = config.dev_config

            conversation_cache = CacheFactory.conversation_cache(
                ols_config.conversation_cache  # type: ignore
            )
    except Exception as e:
        print(f"Failed to load config file {config_file}: {e!s}")
        print(traceback.format_exc())
        raise


def init_query_filter() -> None:
    """Initialize question filter."""
    global query_redactor
    query_redactor = QueryFilter()


def init_vector_index() -> None:
    """Initialize vector index."""
    global rag_index
    rag_index = IndexLoader(ols_config.reference_content).vector_index
