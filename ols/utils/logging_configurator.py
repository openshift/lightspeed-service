"""Logging utilities."""

import logging.config

from ols.app.models.config import LoggingConfig


def configure_logging(logging_config: LoggingConfig) -> None:
    """Configure application logging according to the configuration."""
    log_msg_fmt = (
        "%(asctime)s [%(name)s:%(filename)s:%(lineno)d] %(levelname)s: %(message)s"
    )
    log_config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "loggers": {
            # needs to be root (and not ""), otherwise external libs
            # won't be affected
            "root": {
                "level": logging_config.lib_log_level,
                "handlers": ["console"],
            },
            "ols": {
                "level": logging_config.app_log_level,
                "handlers": ["console"],
                "propagate": False,  # don't propagate to root logger
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "formatters": {
            "standard": {"format": log_msg_fmt},
        },
    }

    logging.config.dictConfig(log_config_dict)
