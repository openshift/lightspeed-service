"""Configuration loader."""

import os
import traceback
from io import TextIOBase

import yaml

import ols.app.models.config as config_model
from ols.src.rag_index.index_loader import IndexLoader
from ols.utils.query_filter import QueryFilter


def load_empty_config() -> None:
    """Initialize empty configuration."""
    return config_model.Config()


def load_config_from_stream(stream: TextIOBase) -> config_model.Config:
    """Load configuration from a YAML stream."""
    data = yaml.safe_load(stream)
    loaded_config = config_model.Config(data)
    loaded_config.validate_yaml()
    return loaded_config


def load_config(config_file: str) -> None:
    """Load configuration from a YAML file."""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return load_config_from_stream(f)
    except Exception as e:
        print(f"Failed to load config file {config_file}: {e!s}")
        print(traceback.format_exc())
        raise


# TODO: move to ols_config or resolve as singleton
def init_query_filter() -> None:
    """Initialize question filter."""
    return QueryFilter()


def init_vector_index(ols_config) -> None:
    """Initialize vector index."""
    return IndexLoader(ols_config.reference_content).vector_index


# this is the config that should be used in the code
cfg_file = os.environ.get("OLS_CONFIG_FILE", "olsconfig.yaml")
config = load_config(cfg_file)
