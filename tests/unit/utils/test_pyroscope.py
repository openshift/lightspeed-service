"""Unit test for the pyroscope utility."""

import logging
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import RequestException

from ols.utils.pyroscope import (
    start_with_pyroscope_enabled,
)


@pytest.fixture
def mock_config():
    """Fixture for mock configuration."""
    mock_cfg = MagicMock()
    mock_cfg.dev_config.pyroscope_url = "http://mock-pyroscope-url"
    mock_cfg.dev_config.run_on_localhost = True
    mock_cfg.dev_config.disable_tls = True
    mock_cfg.ols_config.logging_config.uvicorn_log_level = logging.INFO
    mock_cfg.ols_config.max_workers = 1
    mock_cfg.ols_config.tls_config.tls_key_path = None
    mock_cfg.ols_config.tls_config.tls_certificate_path = None
    mock_cfg.ols_config.tls_config.tls_key_password = None
    mock_cfg.ols_config.tls_security_profile = None
    mock_cfg.rag_index = lambda: None
    return mock_cfg


@pytest.fixture
def mock_logger():
    """Fixture for mock logger."""
    return MagicMock()


def test_pyroscope_server_reachable(mock_config, mock_logger):
    """Test that Pyroscope starts when the server is reachable."""
    with (
        patch("ols.utils.pyroscope.requests.get") as mock_get,
        patch("ols.runners.uvicorn.uvicorn.run") as mock_run,
        patch("ols.utils.pyroscope.threading.Thread") as mock_thread,
        patch.dict("sys.modules", {"pyroscope": MagicMock()}) as mock_pyroscope_module,
    ):
        mock_pyroscope = mock_pyroscope_module["pyroscope"]
        mock_pyroscope.configure = MagicMock()

        mock_get.return_value.status_code = 200

        start_with_pyroscope_enabled(mock_config, mock_logger)

        mock_logger.info.assert_any_call(
            "Pyroscope server is reachable at %s", mock_config.dev_config.pyroscope_url
        )
        mock_pyroscope.configure.assert_called_once()
        mock_run.assert_called_once()
        mock_thread.assert_called_once_with(target=mock_config.rag_index)


def test_pyroscope_server_unreachable(mock_config, mock_logger):
    """Test that Pyroscope logs a failure when the server is unreachable."""
    with patch("ols.utils.pyroscope.requests.get") as mock_get:
        mock_get.return_value.status_code = 500

        start_with_pyroscope_enabled(mock_config, mock_logger)

        mock_logger.info.assert_any_call(
            "Failed to reach Pyroscope server. Status code: %d", 500
        )


def test_pyroscope_connection_error(mock_config, mock_logger):
    """Test that Pyroscope handles connection errors gracefully."""
    with patch(
        "ols.utils.pyroscope.requests.get",
        side_effect=RequestException("Connection error"),
    ):
        start_with_pyroscope_enabled(mock_config, mock_logger)

        mock_logger.info.assert_any_call(
            "Error connecting to Pyroscope server: %s", "Connection error"
        )
