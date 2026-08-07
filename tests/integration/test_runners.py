"""Integration tests for runners."""

import ssl
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from ols import config, constants
from ols.runners.quota_scheduler import start_quota_scheduler
from ols.runners.uvicorn import start_uvicorn
from ols.utils.ssl import split_ciphers

MINIMAL_CONFIG_FILE = "tests/config/valid_config.yaml"
CORRECT_CONFIG_FILE = "tests/config/config_for_integration_tests.yaml"
QUOTA_LIMITERS_CONFIG_FILE = (
    "tests/config/config_for_integration_tests_quota_limiters.yaml"
)


def _run_start_uvicorn_and_assert(
    *,
    host: str,
    port: int,
    ssl_keyfile,
    ssl_certfile,
    ssl_keyfile_password,
    ssl_ciphers: str,
    min_tls_version,
    access_log: bool,
    expected_ciphersuites=None,
) -> None:
    """Call start_uvicorn with mocked uvicorn internals and assert expectations."""
    fake_uvicorn_config = SimpleNamespace(
        ssl=SimpleNamespace(minimum_version=None, set_ciphersuites=Mock()),
        loaded=False,
    )
    fake_uvicorn_config.load = Mock(
        side_effect=lambda: setattr(fake_uvicorn_config, "loaded", True)
    )
    fake_server = SimpleNamespace(run=Mock())

    with (
        patch("ols.runners.uvicorn.uvicorn.Config") as mocked_config,
        patch("ols.runners.uvicorn.uvicorn.Server") as mocked_server,
    ):
        mocked_config.return_value = fake_uvicorn_config
        mocked_server.return_value = fake_server
        start_uvicorn(config)

        mocked_config.assert_called_once_with(
            "ols.app.main:app",
            host=host,
            port=port,
            workers=1,
            log_level=30,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            ssl_keyfile_password=ssl_keyfile_password,
            ssl_version=ssl.PROTOCOL_TLS_SERVER,
            ssl_ciphers=ssl_ciphers,
            access_log=access_log,
        )
        assert fake_uvicorn_config.loaded is True
        assert fake_uvicorn_config.ssl.minimum_version == min_tls_version
        if expected_ciphersuites is not None and hasattr(
            ssl.SSLContext, "set_ciphersuites"
        ):
            fake_uvicorn_config.ssl.set_ciphersuites.assert_called_once_with(
                expected_ciphersuites
            )
        mocked_server.assert_called_once_with(fake_uvicorn_config)
        fake_server.run.assert_called_once_with()


def test_start_uvicorn_minimal_setup():
    """Test the function to start Uvicorn server."""
    config.reload_from_yaml_file(MINIMAL_CONFIG_FILE)
    default_ciphers = split_ciphers(constants.DEFAULT_SSL_CIPHERS)

    _run_start_uvicorn_and_assert(
        host="0.0.0.0",  # noqa: S104
        port=8080,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_keyfile_password=None,
        ssl_ciphers=default_ciphers.tls12 or "DEFAULT",
        expected_ciphersuites=default_ciphers.tls13,
        min_tls_version=None,
        access_log=False,
    )


def test_start_uvicorn_full_setup():
    """Test the function to start Uvicorn server."""
    config.reload_from_yaml_file(CORRECT_CONFIG_FILE)
    custom_ciphers = split_ciphers(
        "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256, TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"
    )

    _run_start_uvicorn_and_assert(
        host="0.0.0.0",  # noqa: S104
        port=8080,
        ssl_keyfile="tests/config/key",
        ssl_certfile="tests/config/empty_cert.crt",
        ssl_keyfile_password="* this is password *",  # noqa: S106
        ssl_ciphers=custom_ciphers.tls12 or "DEFAULT",
        expected_ciphersuites=custom_ciphers.tls13,
        min_tls_version=ssl.TLSVersion.TLSv1_3,
        access_log=False,
    )


@pytest.mark.filterwarnings("ignore")
def test_start_quota_scheduler():
    """Test the function to start Quota scheduler."""
    config.reload_from_yaml_file(QUOTA_LIMITERS_CONFIG_FILE)
    with (
        patch("ols.runners.quota_scheduler.sleep", side_effect=Exception()),
        patch("psycopg2.connect"),
    ):
        # just try to enter the endless loop
        start_quota_scheduler(config)
