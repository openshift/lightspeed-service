"""Unit tests for gradioUI class."""

import logging
from unittest.mock import patch

import requests

from ols.src.ui.gradio_ui import GradioUI
from ols.utils.logging import LoggingConfig, configure_logging


def setup_logging(caplog):
    """Set up logging and capturing log messsages."""
    logging_config = LoggingConfig(
        **{"app_log_level": "info"},
    )
    configure_logging(logging_config)
    logger = logging.getLogger("ols")
    logger.handlers = [caplog.handler]  # add caplog handler to logger


def test_gradio_ui_constructor():
    """Test if all attributes are setup correctly by constructor."""
    url = "locahost:8080"
    conversation_id = "01234567-89ab-cdef-0123-456789abcdef"

    ui = GradioUI(ols_url=url, conversation_id=conversation_id)
    assert ui is not None
    assert ui.ols_url == url
    assert ui.conversation_id == conversation_id


def test_chat_ui_handler_ok_response(caplog):
    """Test the UI handler for proper REST API response."""
    setup_logging(caplog)

    ok_response = requests.Response()
    ok_response.status_code = requests.codes.ok
    ok_response.json = lambda: {"response": "this is response"}

    with patch("ols.src.ui.gradio_ui.requests.post", return_value=ok_response):
        ui = GradioUI()
        ret = ui.chat_ui("prompt", None, False)
        assert ret == "this is response"

    captured_out = caplog.text
    assert "Using history: False" in captured_out
    assert "Ignoring conversation history" in captured_out


def test_chat_ui_handler_use_history_enabled(caplog):
    """Test the UI handler for proper REST API response when history is enabled."""
    setup_logging(caplog)

    ok_response = requests.Response()
    ok_response.status_code = requests.codes.ok
    ok_response.json = lambda: {"response": "this is response"}

    with patch("ols.src.ui.gradio_ui.requests.post", return_value=ok_response):
        ui = GradioUI()
        ret = ui.chat_ui("prompt", None, True)
        assert ret == "this is response"

    captured_out = caplog.text
    assert "Using history: True" in captured_out
    assert "Ignoring conversation history" not in captured_out


def test_chat_ui_handler_use_with_conversation_id(caplog):
    """Test the UI handler for proper REST API response when history is enabled."""
    ok_response = requests.Response()
    ok_response.status_code = requests.codes.ok
    ok_response.json = lambda: {"response": "this is response"}

    conversation_id = "01234567-89ab-cdef-0123-456789abcdef"
    with patch("ols.src.ui.gradio_ui.requests.post", return_value=ok_response):
        ui = GradioUI(conversation_id=conversation_id)
        ret = ui.chat_ui("prompt", None, True)
        assert ret == "this is response"

    captured_out = caplog.text
    assert "Ignoring conversation history" not in captured_out
    assert f"Using conversation ID: {conversation_id}" in captured_out


def test_chat_ui_handler_use_with_provider(caplog):
    """Test the UI handler for proper REST API response when provider is setup."""
    ok_response = requests.Response()
    ok_response.status_code = requests.codes.ok
    ok_response.json = lambda: {"response": "this is response"}

    with patch("ols.src.ui.gradio_ui.requests.post", return_value=ok_response):
        ui = GradioUI()
        ret = ui.chat_ui("prompt", None, True, provider="PROVIDER")
        assert ret == "this is response"

    captured_out = caplog.text
    assert "Using provider: PROVIDER" in captured_out


def test_chat_ui_handler_use_with_model(caplog):
    """Test the UI handler for proper REST API response when model is setup."""
    ok_response = requests.Response()
    ok_response.status_code = requests.codes.ok
    ok_response.json = lambda: {"response": "this is response"}

    with patch("ols.src.ui.gradio_ui.requests.post", return_value=ok_response):
        ui = GradioUI()
        ret = ui.chat_ui("prompt", None, True, model="MODEL")
        assert ret == "this is response"

    captured_out = caplog.text
    assert "Using model: MODEL" in captured_out


def test_chat_ui_handler_bad_http_code():
    """Test the UI handler for REST API response that is not OK."""
    bad_response = requests.Response()
    bad_response.status_code = requests.codes.internal_server_error

    with patch("ols.src.ui.gradio_ui.requests.post", return_value=bad_response):
        ui = GradioUI()
        ret = ui.chat_ui("prompt", None, False)
        assert "Sorry, an error occurred" in ret


def test_chat_ui_handler_error_handling():
    """Test error handling in UI handler."""
    with patch(
        "ols.src.ui.gradio_ui.requests.post", side_effect=requests.exceptions.HTTPError
    ):
        ui = GradioUI()
        ret = ui.chat_ui("prompt", None, False)
        assert "An error occurred" in ret
