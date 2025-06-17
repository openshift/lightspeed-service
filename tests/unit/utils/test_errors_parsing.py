"""Unit Test for the errors parsing class."""

from genai.exceptions import ApiResponseException
from httpx import Request, Response
from openai import BadRequestError

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
    assert error_message == "Exception"
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
    assert error_message == message
    assert cause == "Exception"


def test_parse_generic_llm_error_on_api_response_exception_without_info():
    """Test the parse_generic_llm_error function when ApiResponseException is passed."""
    expected_status_code = BAD_REQUEST_STATUS_CODE
    error = ApiResponseException(
        response={
            "error": "",
            "message": "",
            "extensions": {},
            "status_code": BAD_REQUEST_STATUS_CODE,
        },
        message=None,
    )

    # try to parse the exception
    status_code, error_message, cause = errors_parsing.parse_generic_llm_error(error)

    # check the parsed error
    assert status_code == expected_status_code
    expected = """Server Error
{
  "error": "",
  "extensions": {
    "code": "INVALID_INPUT",
    "state": null
  },
  "message": "",
  "status_code": 400
}"""
    assert error_message == expected
    assert cause == expected


def test_parse_generic_llm_error_on_api_response_exception_with_info():
    """Test the parse_generic_llm_error function when ApiResponseException is passed."""
    expected_status_code = BAD_REQUEST_STATUS_CODE
    error = ApiResponseException(
        response={
            "error": "this is error",
            "message": "this is error message",
            "extensions": {},
            "status_code": BAD_REQUEST_STATUS_CODE,
        },
        message=None,
    )

    # try to parse the exception
    status_code, error_message, cause = errors_parsing.parse_generic_llm_error(error)

    # check the parsed error
    assert status_code == expected_status_code
    expected = """Server Error
{
  "error": "this is error",
  "extensions": {
    "code": "INVALID_INPUT",
    "state": null
  },
  "message": "this is error message",
  "status_code": 400
}"""
    assert error_message == expected
    assert cause == expected


def test_parse_generic_llm_error_on_api_response_exception_with_message():
    """Test the parse_generic_llm_error function when ApiResponseException is passed."""
    expected_status_code = BAD_REQUEST_STATUS_CODE
    message = "This is proper error message"

    error = ApiResponseException(
        message=message,
        response={
            "message": "this is error message",
            "error": "this is error",
            "extensions": {},
            "status_code": BAD_REQUEST_STATUS_CODE,
        },
    )

    # try to parse the exception
    status_code, error_message, cause = errors_parsing.parse_generic_llm_error(error)

    # check the parsed error
    assert status_code == expected_status_code
    expected = (
        message
        + """
{
  "error": "this is error",
  "extensions": {
    "code": "INVALID_INPUT",
    "state": null
  },
  "message": "this is error message",
  "status_code": 400
}"""
    )
    assert error_message == expected
    assert cause == expected


def test_parse_generic_llm_error_on_api_response_exception_without_extensions_state():
    """Test the parse_generic_llm_error function when ApiResponseException is passed."""
    expected_status_code = BAD_REQUEST_STATUS_CODE
    message = "This is proper error message"

    error = ApiResponseException(
        message=message,
        response={
            "message": "this is error message",
            "error": "this is error",
            "extensions": {},
            "status_code": BAD_REQUEST_STATUS_CODE,
        },
    )
    # cleanup extensions state
    error.response.extensions.state = None

    # try to parse the exception
    status_code, error_message, cause = errors_parsing.parse_generic_llm_error(error)

    # check the parsed error
    assert status_code == expected_status_code
    expected = (
        message
        + """
{
  "error": "this is error",
  "extensions": {
    "code": "INVALID_INPUT",
    "state": null
  },
  "message": "this is error message",
  "status_code": 400
}"""
    )
    assert error_message == expected
    assert cause == expected


def test_parse_generic_llm_error_on_api_response_exception_with_extensions_state():
    """Test the parse_generic_llm_error function when ApiResponseException is passed."""
    expected_status_code = BAD_REQUEST_STATUS_CODE
    message = "This is proper error message"

    error = ApiResponseException(
        message=message,
        response={
            "message": "this is error message",
            "error": "this is error",
            "extensions": {
                "code": "INVALID_INPUT",
                "state": {"message": "state message"},
            },
            "status_code": BAD_REQUEST_STATUS_CODE,
        },
    )
    # cleanup extensions state
    error.response.extensions.state = None

    # try to parse the exception
    status_code, error_message, cause = errors_parsing.parse_generic_llm_error(error)

    # check the parsed error
    assert status_code == expected_status_code
    expected = (
        message
        + """
{
  "error": "this is error",
  "extensions": {
    "code": "INVALID_INPUT",
    "state": {
      "message": "state message"
    }
  },
  "message": "this is error message",
  "status_code": 400
}"""
    )
    assert error_message == expected
    assert cause == expected


def test_parse_generic_llm_error_on_unknown_exception():
    """Test the parse_generic_llm_error function when unknown exception is passed."""
    # generic exception
    error = Exception("Exception")

    # try to parse the exception
    status_code, error_message, cause = errors_parsing.parse_generic_llm_error(error)

    # check the parsed error
    assert status_code == errors_parsing.DEFAULT_STATUS_CODE
    assert error_message == errors_parsing.DEFAULT_ERROR_MESSAGE
    assert cause == "Exception"


def test_parse_generic_llm_error_on_unknown_exception_without_message():
    """Test the parse_generic_llm_error function when unknown exception is passed."""
    # generic exception
    error = Exception()

    # try to parse the exception
    status_code, error_message, cause = errors_parsing.parse_generic_llm_error(error)

    # check the parsed error
    assert status_code == errors_parsing.DEFAULT_STATUS_CODE
    assert error_message == errors_parsing.DEFAULT_ERROR_MESSAGE
    assert cause == ""


def test_handle_known_errors():
    """Test the handle_known_errors function."""
    # generic exception
    known_response = "Maximum context length exceeded"
    unknown_response = "This is unknown response"
    cause = "cause"

    # known error
    handled_response, handled_cause = errors_parsing.handle_known_errors(
        known_response, cause
    )
    assert handled_response == errors_parsing.PROMPT_TOO_LONG_ERROR_MSG
    assert handled_cause == "cause"

    # unknown error
    handled_response, handled_cause = errors_parsing.handle_known_errors(
        unknown_response, cause
    )
    assert handled_response == unknown_response
    assert handled_cause == cause
