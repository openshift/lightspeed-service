"""Unit tests for gradioUI class."""

from unittest.mock import patch

import requests

from ols.src.ui.gradio_ui import gradioUI


def test_gradio_ui_constructor():
    """Test if all attributes are setup correctly by constructor."""
    URL = "locahost:8080"
    conversation_id = 1234

    ui = gradioUI(ols_url=URL, conversation_id=conversation_id, logger=None)
    assert ui is not None
    assert ui.ols_url == URL
    assert ui.conversation_id == conversation_id
    assert ui.logger is not None


def test_chat_ui_handler_ok_response():
    """Test the UI handler for proper REST API response."""
    okResponse = requests.Response()
    okResponse.status_code = requests.codes.ok
    okResponse.json = lambda: {"response": "this is response"}

    with patch("ols.src.ui.gradio_ui.requests.post", return_value=okResponse):
        ui = gradioUI()
        ret = ui.chat_ui("prompt", None, False)
        assert ret == "this is response"


def test_chat_ui_handler_bad_http_code():
    """Test the UI handler for REST API response that is not OK."""
    badResponse = requests.Response()
    badResponse.status_code = requests.codes.internal_server_error

    with patch("ols.src.ui.gradio_ui.requests.post", return_value=badResponse):
        ui = gradioUI()
        ret = ui.chat_ui("prompt", None, False)
        assert "Sorry, an error occurred" in ret


def test_chat_ui_handler_error_handling():
    """Test error handling in UI handler."""
    with patch(
        "ols.src.ui.gradio_ui.requests.post", side_effect=requests.exceptions.HTTPError
    ):
        ui = gradioUI()
        ret = ui.chat_ui("prompt", None, False)
        assert "An error occurred" in ret
