"""Unit Test for the errors parsing class."""

from unittest.mock import patch

import httpx
from httpx import Request, Response
from openai import APIConnectionError, APITimeoutError, BadRequestError

from ols.utils import errors_parsing

BAD_REQUEST_STATUS_CODE = 400


def test_parse_generic_llm_error_on_bad_request_error_without_info():
    """Test the parse_generic_llm_error function when BadRequestError is passed."""
    expected_status_code = BAD_REQUEST_STATUS_CODE
    response = Response(
        status_code=expected_status_code,
        request=Request(method="GET", url="http://foo.com"),
    )
    error = BadRequestError("Exception", response=response, body=None)

    # try to parse the exception
    status_code, error_message, cause = errors_parsing.parse_generic_llm_error(error)

    # check the parsed error
    assert status_code == expected_status_code
    assert error_message == f"{errors_parsing._LLM_BACKEND_PREFIX} Exception"
    assert cause == "Exception"


def test_parse_generic_llm_error_on_bad_request_error_with_info():
    """Test the parse_generic_llm_error function when BadRequestError is passed."""
    expected_status_code = BAD_REQUEST_STATUS_CODE
    message = "Exception message"
    response = Response(
        status_code=expected_status_code,
        request=Request(method="GET", url="http://foo.com"),
    )
    error = BadRequestError("Exception", response=response, body={"message": message})

    # try to parse the exception
    status_code, error_message, cause = errors_parsing.parse_generic_llm_error(error)

    # check the parsed error
    assert status_code == expected_status_code
    assert error_message == f"{errors_parsing._LLM_BACKEND_PREFIX} {message}"
    assert cause == "Exception"


def test_parse_generic_llm_error_on_unknown_exception():
    """Test the parse_generic_llm_error function when unknown exception is passed."""
    # generic exception
    error = Exception("Exception")

    # try to parse the exception
    status_code, error_message, cause = errors_parsing.parse_generic_llm_error(error)

    # check the parsed error
    assert status_code == errors_parsing.DEFAULT_STATUS_CODE
    assert (
        error_message
        == f"{errors_parsing._LLM_BACKEND_PREFIX} {errors_parsing.DEFAULT_ERROR_MESSAGE}"
    )
    assert cause == "Exception"


def test_parse_generic_llm_error_on_unknown_exception_without_message():
    """Test the parse_generic_llm_error function when unknown exception is passed."""
    # generic exception
    error = Exception()

    # try to parse the exception
    status_code, error_message, cause = errors_parsing.parse_generic_llm_error(error)

    # check the parsed error
    assert status_code == errors_parsing.DEFAULT_STATUS_CODE
    assert (
        error_message
        == f"{errors_parsing._LLM_BACKEND_PREFIX} {errors_parsing.DEFAULT_ERROR_MESSAGE}"
    )
    assert cause == ""


def test_handle_known_errors():
    """Test the handle_known_errors function."""
    # generic exception
    known_response = "Maximum context length exceeded"
    unknown_response = "This is unknown response"
    cause = "cause"

    # known error - prompt too long
    handled_response, handled_cause = errors_parsing.handle_known_errors(
        known_response, cause
    )
    assert (
        handled_response
        == f"{errors_parsing._LLM_BACKEND_PREFIX} {errors_parsing.PROMPT_TOO_LONG_ERROR_MSG}"
    )
    assert handled_cause == "cause"

    # known error - prompt too long with tools
    with patch("ols.config.mcp_servers.servers", new=["server"]):
        handled_response, handled_cause = errors_parsing.handle_known_errors(
            known_response, cause
        )
    expected = (
        f"{errors_parsing._LLM_BACKEND_PREFIX}"
        f" {errors_parsing.PROMPT_TOO_LONG_WITH_TOOLS_ERROR_MSG}"
    )
    assert handled_response == expected
    assert handled_cause == "cause"

    # unknown error
    handled_response, handled_cause = errors_parsing.handle_known_errors(
        unknown_response, cause
    )
    assert handled_response == unknown_response
    assert handled_cause == cause


def test_parse_generic_llm_error_on_api_connection_error():
    """Test network error prefix for OpenAI APIConnectionError."""
    error = APIConnectionError(request=Request(method="POST", url="http://llm.svc"))

    status_code, error_message, _cause = errors_parsing.parse_generic_llm_error(error)

    assert status_code == 502
    assert error_message.startswith(errors_parsing._NETWORK_PREFIX)
    assert errors_parsing.NETWORK_ERROR_MSG in error_message


def test_parse_generic_llm_error_on_api_timeout_error():
    """Test network error prefix for OpenAI APITimeoutError."""
    error = APITimeoutError(request=Request(method="POST", url="http://llm.svc"))

    status_code, error_message, _cause = errors_parsing.parse_generic_llm_error(error)

    assert status_code == 502
    assert error_message.startswith(errors_parsing._NETWORK_PREFIX)
    assert errors_parsing.NETWORK_ERROR_MSG in error_message


def test_parse_generic_llm_error_on_httpx_connect_error():
    """Test network error prefix for httpx ConnectError."""
    error = httpx.ConnectError("Connection refused")

    status_code, error_message, cause = errors_parsing.parse_generic_llm_error(error)

    assert status_code == 502
    assert error_message.startswith(errors_parsing._NETWORK_PREFIX)
    assert "Connection refused" in cause


def test_parse_generic_llm_error_on_httpx_connect_timeout():
    """Test network error prefix for httpx ConnectTimeout."""
    error = httpx.ConnectTimeout("Timed out")

    status_code, error_message, cause = errors_parsing.parse_generic_llm_error(error)

    assert status_code == 502
    assert error_message.startswith(errors_parsing._NETWORK_PREFIX)
    assert "Timed out" in cause
