"""Configuration loader."""

import traceback

import yaml

import ols.app.models.config as config_model
from ols.src.cache.cache_factory import CacheFactory
from ols.utils.logger import Logger

config = None
ols_config = None
llm_config = None
default_logger = None
conversation_cache = None


def init_empty_config():
    """Initialize empty configuration."""
    global config
    global ols_config
    global llm_config
    config = config_model.Config()
    ols_config = config_model.OLSConfig()
    llm_config = config_model.LLMProviders()


def load_config_from_stream(stream) -> config_model.Config:
    """Load configuration from a YAML stream."""
    data = yaml.safe_load(stream)
    c = config_model.Config(data)
    c.validate_yaml()
    return c


def init_config(config_file):
    """Load configuration from a YAML file."""
    global config
    global ols_config
    global llm_config
    global default_logger
    global conversation_cache

    try:
        with open(config_file, "r") as f:
            config = load_config_from_stream(f)
            ols_config = config.ols_config
            llm_config = config.llm_providers

            default_logger = Logger(
                logger_name="default", log_level=ols_config.logger_config.default_level
            ).logger
            conversation_cache = CacheFactory.conversation_cache(
                ols_config.conversation_cache
            )
    except Exception as e:
        print(f"Failed to load config file {config_file}: {e!s}")
        print(traceback.format_exc())
        raise e
