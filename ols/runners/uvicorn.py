"""Uvicorn runner."""

import logging

import uvicorn

from ols.utils import ssl
from ols.utils.config import AppConfig

logger: logging.Logger = logging.getLogger(__name__)


def start_uvicorn(config: AppConfig) -> None:
    """Start Uvicorn-based REST API service."""
    logger.info("Starting Uvicorn")

    # use workers=1 so config loaded can be accessed from other modules
    host = (
        "localhost"
        if config.dev_config.run_on_localhost
        else "0.0.0.0"  # noqa: S104 # nosec: B104
    )
    port = (
        config.dev_config.uvicorn_port_number
        if config.dev_config.uvicorn_port_number
        else 8080 if config.dev_config.disable_tls else 8443
    )
    log_level = config.ols_config.logging_config.uvicorn_log_level

    # The tls fields can be None, which means we will pass those values as None to uvicorn.run
    ssl_keyfile = config.ols_config.tls_config.tls_key_path
    ssl_certfile = config.ols_config.tls_config.tls_certificate_path
    ssl_keyfile_password = config.ols_config.tls_config.tls_key_password

    # setup SSL version and allowed SSL ciphers based on service configuration
    # when TLS security profile is not specified, default values will be used
    # that default values are based on default SSL package settings
    sec_profile = config.ols_config.tls_security_profile
    ssl_version = ssl.get_ssl_version(sec_profile)
    ssl_ciphers = ssl.get_ciphers(sec_profile)

    uvicorn.run(
        "ols.app.main:app",
        host=host,
        port=port,
        workers=config.ols_config.max_workers,
        log_level=log_level,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        ssl_keyfile_password=ssl_keyfile_password,
        ssl_version=ssl_version,
        ssl_ciphers=ssl_ciphers,
        access_log=log_level < logging.INFO,
    )
