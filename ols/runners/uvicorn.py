"""Uvicorn runner."""

import logging

import uvicorn

from ols.utils import ssl as ssl_utils
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
    port = config.dev_config.uvicorn_port_number or (
        8080 if config.dev_config.disable_tls else 8443
    )
    log_level = config.ols_config.logging_config.uvicorn_log_level

    # The tls fields can be None, which means we will pass those values through to Uvicorn.
    ssl_keyfile = config.ols_config.tls_config.tls_key_path
    ssl_certfile = config.ols_config.tls_config.tls_certificate_path
    ssl_keyfile_password = config.ols_config.tls_config.tls_key_password

    # setup SSL version and allowed SSL ciphers based on service configuration
    # when TLS security profile is not specified, default values will be used
    # that default values are based on default SSL package settings
    sec_profile = config.ols_config.tls_security_profile
    ssl_version = ssl_utils.get_ssl_version(sec_profile)
    min_tls_version = ssl_utils.get_min_tls_version(sec_profile)
    ssl_ciphers_str = ssl_utils.get_ciphers(sec_profile)
    ciphers = ssl_utils.split_ciphers(ssl_ciphers_str)

    uvicorn_config = uvicorn.Config(
        "ols.app.main:app",
        host=host,
        port=port,
        workers=config.ols_config.max_workers,
        log_level=log_level,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        ssl_keyfile_password=ssl_keyfile_password,
        ssl_version=ssl_version,
        ssl_ciphers=ciphers.tls12 or "DEFAULT",
        access_log=log_level < logging.INFO,
    )
    uvicorn_config.load()
    if uvicorn_config.ssl is not None:
        if min_tls_version is not None:
            uvicorn_config.ssl.minimum_version = min_tls_version
        if ciphers.tls13 is not None and hasattr(
            uvicorn_config.ssl, "set_ciphersuites"
        ):
            uvicorn_config.ssl.set_ciphersuites(ciphers.tls13)

    server = uvicorn.Server(uvicorn_config)
    server.run()
