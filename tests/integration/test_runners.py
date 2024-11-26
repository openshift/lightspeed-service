"""Integration tests for runners."""

import ssl
from unittest.mock import patch

from ols import config
from ols.runners.uvicorn import start_uvicorn

MINIMAL_CONFIG_FILE = "tests/config/valid_config.yaml"
CORRECT_CONFIG_FILE = "tests/config/config_for_integration_tests.yaml"


@patch("uvicorn.run")
def test_start_uvicorn_minimal_setup(mocked_runner):
    """Test the function to start Uvicorn server."""
    config.reload_from_yaml_file(MINIMAL_CONFIG_FILE)
    start_uvicorn(config)
    mocked_runner.assert_called_once_with(
        "ols.app.main:app",
        host="0.0.0.0",  # noqa: S104
        port=8080,
        workers=1,
        log_level=30,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_keyfile_password=None,
        ssl_version=ssl.PROTOCOL_TLS_SERVER,
        ssl_ciphers="TLSv1",
        access_log=False,
    )


@patch("uvicorn.run")
def test_start_uvicorn_full_setup(mocked_runner):
    """Test the function to start Uvicorn server."""
    config.reload_from_yaml_file(CORRECT_CONFIG_FILE)
    start_uvicorn(config)
    mocked_runner.assert_called_once_with(
        "ols.app.main:app",
        host="0.0.0.0",  # noqa: S104
        port=8080,
        workers=1,
        log_level=30,
        ssl_keyfile="tests/config/key",
        ssl_certfile="tests/config/empty_cert.crt",
        ssl_keyfile_password="* this is password *",  # noqa: S106
        ssl_version=ssl.TLSVersion.TLSv1_3,
        ssl_ciphers="TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256, TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
        access_log=False,
    )
