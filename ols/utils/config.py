"""Configuration loader."""

import logging
import os

import ols.app.models.config as config_model
from ols.src import constants
from ols.src.cache.cache_factory import CacheFactory
from ols.utils.logger import Logger

ols_config = None
llm_config = None
default_logger = None
feedback_logger = None
conversation_cache = None


def load_empty_config() -> None:
    """Load empty configuration."""
    global ols_config
    global llm_config
    ols_config = config_model.OLSConfig()
    llm_config = config_model.LLMConfig()


def load_config_from_env() -> None:
    """Load configuration from environment variables."""
    global ols_config
    global llm_config
    global default_logger
    global conversation_cache

    ols_config = config_model.OLSConfig()

    logger_config = config_model.LoggerConfig()
    ols_config.logger_config = logger_config

    logger_config.default_level = os.getenv(
        "LOG_LEVEL", logging.getLevelName(logging.INFO)
    )
    default_logger = Logger(
        logger_name="default",
        log_level=logger_config.default_level,
    ).logger

    ols_config.enable_debug_ui = (
        True if os.getenv("ENABLE_DEV_UI", "False").lower() == "true" else False
    )

    ols_config.default_provider = os.getenv("DEFAULT_PROVIDER", constants.PROVIDER_BAM)
    ols_config.default_model = os.getenv("DEFAULT_MODEL", constants.GRANITE_13B_CHAT_V1)
    ols_config.classifier_provider = os.getenv(
        "CLASSIFIER_PROVIDER", ols_config.default_provider
    )
    ols_config.classifier_model = os.getenv(
        "CLASSIFIER_MODEL", ols_config.default_model
    )
    ols_config.summarizer_provider = os.getenv(
        "SUMMARIZER_PROVIDER", ols_config.default_provider
    )
    ols_config.summarizer_model = os.getenv(
        "SUMMARIZER_MODEL", ols_config.default_model
    )
    ols_config.validator_provider = os.getenv(
        "VALIDATOR_PROVIDER", ols_config.default_provider
    )
    ols_config.validator_model = os.getenv("VALIDATOR_MODEL", ols_config.default_model)
    ols_config.yaml_provider = os.getenv("YAML_PROVIDER", ols_config.default_provider)
    ols_config.yaml_model = os.getenv("YAML_MODEL", ols_config.default_model)

    cache_config = config_model.ConversationCacheConfig()
    ols_config.conversation_cache = cache_config
    cache_config.type = os.getenv(
        "OLS_CONVERSATION_CACHE", constants.IN_MEMORY_CACHE
    ).lower()
    if cache_config.type == constants.REDIS_CACHE:
        redis_config = config_model.RedisConfig()
        cache_config.redis = redis_config
        redis_config.host = os.getenv("REDIS_CACHE_HOST", constants.REDIS_CACHE_HOST)
        redis_config.port = int(
            os.getenv("REDIS_CACHE_PORT", constants.REDIS_CACHE_PORT)
        )
        redis_config.max_memory = constants.REDIS_CACHE_MAX_MEMORY
        redis_config.max_memory_policy = constants.REDIS_CACHE_MAX_MEMORY_POLICY
    elif cache_config.type == constants.IN_MEMORY_CACHE:
        memory_config = config_model.MemoryConfig({})
        cache_config.memory = memory_config
        memory_config.max_entries = int(
            os.getenv("MEMORY_CACHE_MAX_ENTRIES", constants.IN_MEMORY_CACHE_MAX_ENTRIES)
        )
    else:
        raise Exception(f"Invalid cache type: {cache_config.type}")
    conversation_cache = CacheFactory.conversation_cache(cache_config)

    llm_config = config_model.LLMConfig()
    if os.getenv("BAM_API_KEY", None) is not None:
        bam_provider = config_model.ProviderConfig()
        bam_provider.name = constants.PROVIDER_BAM
        bam_provider.credentials = os.getenv("BAM_API_KEY", None)
        bam_provider.url = os.getenv("BAM_API_URL", "https://bam-api.res.ibm.com")

        for model in [
            constants.GRANITE_13B_CHAT_V1,
            constants.GRANITE_13B_CHAT_V2,
            constants.GRANITE_20B_CODE_INSTRUCT_V1,
        ]:
            m = config_model.ModelConfig()
            m.name = model
            bam_provider.models[m.name] = m
        llm_config.providers[bam_provider.name] = bam_provider

    if os.getenv("OPENAI_API_KEY", None) is not None:
        oai_provider = config_model.ProviderConfig()
        oai_provider.name = constants.PROVIDER_OPENAI
        oai_provider.credentials = os.getenv("OPENAI_API_KEY", None)
        oai_provider.url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")

        for model in [
            constants.GPT35_TURBO_1106,
            constants.GPT35_TURBO,
        ]:
            m = config_model.ModelConfig()
            m.name = model
            oai_provider.models[m.name] = m
        llm_config.providers[oai_provider.name] = oai_provider


# def load_config(config_file):
#     try:
#         with open(config_file, "r") as f:
#             data = yaml.safe_load(f)
#             c=Config(data)
#     except Exception as e:
#         print(f"Failed to load config file {config_file}: {str(e)}")
#         print(traceback.format_exc())
#         exit(1)
#     try:
#         c.validate()
#     except Exception as e:
#         print(f"Failed to validate config file {config_file}: {str(e)}")
#         print(traceback.format_exc())
#         exit(1)
#     global config
#     global default_logger
#     global feedback_logger
#     global conversation_cache
#     config = c
#     default_logger = Logger(logger_name="default", logfile=c.ols_config.logger_config.default_filename,log_level=c.ols_config.logger_config.default_level).logger
#     feedback_logger = Logger(logger_name="feedback", logfile=c.ols_config.logger_config.feedback_filename,log_level=c.ols_config.logger_config.feedback_level).logger

#     conversation_cache = CacheFactory.conversation_cache(c.ols_config.conversation_cache)
#     return c
