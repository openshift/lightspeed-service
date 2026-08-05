"""Unit tests for runners."""

import ssl as stdlib_ssl
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from ols import constants
from ols.app.models.config import Config, TLSSecurityProfile
from ols.runners.uvicorn import start_uvicorn
from ols.utils import tls


@pytest.fixture
def default_config():
    """Fixture providing default configuration."""
    return Config(
        {
            "llm_providers": [],
            "ols_config": {
                "default_provider": "test_default_provider",
                "default_model": "test_default_model",
                "conversation_cache": {
                    "type": "memory",
                    "memory": {
                        "max_entries": 100,
                    },
                },
                "logging_config": {
                    "app_log_level": "error",
                },
                "authentication_config": {"module": "foo"},
            },
            "dev_config": {"disable_tls": "true"},
        }
    )


def _assert_start_uvicorn(
    config: Config,
    *,
    host: str,
    port: int,
    min_tls_version,
    ssl_ciphers,
) -> None:
    """Assert the Uvicorn runner configures and starts the server."""
    fake_uvicorn_config = SimpleNamespace(
        ssl=SimpleNamespace(minimum_version=None),
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
            ssl_keyfile=None,
            ssl_certfile=None,
            ssl_keyfile_password=None,
            ssl_version=constants.DEFAULT_SSL_VERSION,
            ssl_ciphers=ssl_ciphers,
            access_log=False,
        )
        assert fake_uvicorn_config.loaded is True
        assert fake_uvicorn_config.ssl.minimum_version == min_tls_version
        mocked_server.assert_called_once_with(fake_uvicorn_config)
        fake_server.run.assert_called_once_with()


def test_start_uvicorn(default_config):
    """Test the function to start Uvicorn server."""
    _assert_start_uvicorn(
        default_config,
        host="0.0.0.0",  # noqa: S104
        port=8080,
        min_tls_version=None,
        ssl_ciphers=constants.DEFAULT_SSL_CIPHERS,
    )


def test_start_uvicorn_with_tls(default_config):
    """Test the function to start Uvicorn server with TLS enabled."""
    default_config.dev_config.disable_tls = False
    _assert_start_uvicorn(
        default_config,
        host="0.0.0.0",  # noqa: S104
        port=8443,
        min_tls_version=None,
        ssl_ciphers=constants.DEFAULT_SSL_CIPHERS,
    )


def test_start_uvicorn_on_localhost(default_config):
    """Test the function to start Uvicorn server."""
    default_config.dev_config.run_on_localhost = True
    _assert_start_uvicorn(
        default_config,
        host="localhost",
        port=8080,
        min_tls_version=None,
        ssl_ciphers=constants.DEFAULT_SSL_CIPHERS,
    )


def test_start_uvicorn_on_non_default_port(default_config):
    """Test the function to start Uvicorn server on a non-default port."""
    default_config.dev_config.uvicorn_port_number = 8081
    _assert_start_uvicorn(
        default_config,
        host="0.0.0.0",  # noqa: S104
        port=8081,
        min_tls_version=None,
        ssl_ciphers=constants.DEFAULT_SSL_CIPHERS,
    )


@pytest.mark.parametrize(
    "profile_type,min_tls_version",
    [
        ("IntermediateType", stdlib_ssl.TLSVersion.TLSv1_2),
        ("ModernType", stdlib_ssl.TLSVersion.TLSv1_3),
    ],
)
def test_start_uvicorn_applies_min_tls_version(
    default_config, profile_type, min_tls_version
):
    """Test the function to start Uvicorn server with a TLS security profile."""
    default_config.dev_config.disable_tls = False
    default_config.ols_config.tls_security_profile = TLSSecurityProfile(
        {"type": profile_type}
    )
    _assert_start_uvicorn(
        default_config,
        host="0.0.0.0",  # noqa: S104
        port=8443,
        min_tls_version=min_tls_version,
        ssl_ciphers=tls.ciphers_for_tls_profile(profile_type),
    )
