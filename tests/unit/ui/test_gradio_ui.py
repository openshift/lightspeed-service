"""Unit tests for gradioUI class."""

from unittest.mock import patch

import requests

from ols.src.ui.gradio_ui import GradioUI


def test_gradio_ui_constructor():
    """Test if all attributes are setup correctly by constructor."""
    url = "locahost:8080"
    conversation_id = 1234

    ui = GradioUI(ols_url=url, conversation_id=conversation_id)
    assert ui is not None
    assert ui.ols_url == url
    assert ui.conversation_id == conversation_id


def test_chat_ui_handler_ok_response():
    """Test the UI handler for proper REST API response."""
    ok_response = requests.Response()
    ok_response.status_code = requests.codes.ok
    ok_response.json = lambda: {"response": "this is response"}

    with patch("ols.src.ui.gradio_ui.requests.post", return_value=ok_response):
        ui = GradioUI()
        ret = ui.chat_ui("prompt", None, False)
        assert ret == "this is response"


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
