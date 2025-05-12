"""Benchmarks for errors parsing feature."""

from genai.exceptions import ApiResponseException
from httpx import Request, Response
from openai import BadRequestError

from ols.utils import errors_parsing


def test_parse_generic_llm_error_on_bad_request_error_without_info(benchmark):
    """Benchmark the parse_generic_llm_error function when BadRequestError is passed."""
    status_code = 400
    response = Response(
        status_code=status_code,
        request=Request(method="GET", url="http://foo.com"),
    )
    error = BadRequestError("Exception", response=response, body=None)

    # benchmark the function to parse LLM errors
    benchmark(errors_parsing.parse_generic_llm_error, error)


def test_parse_generic_llm_error_on_bad_request_error_with_info(benchmark):
    """Benchmark the parse_generic_llm_error function when BadRequestError is passed."""
    status_code = 400
    message = "Exception message"
    response = Response(
        status_code=status_code,
        request=Request(method="GET", url="http://foo.com"),
    )
    error = BadRequestError("Exception", response=response, body={"message": message})

    # benchmark the function to parse LLM errors
    benchmark(errors_parsing.parse_generic_llm_error, error)


def test_parse_generic_llm_error_on_api_response_exception_without_info(benchmark):
    """Benchmark the parse_generic_llm_error function when ApiResponseException is passed."""
    error = ApiResponseException(
        response={
            "error": "",
            "message": "",
            "extensions": {},
            "status_code": 400,
        },
        message=None,
    )

    # benchmark the function to parse LLM errors
    benchmark(errors_parsing.parse_generic_llm_error, error)


def test_parse_generic_llm_error_on_api_response_exception_with_info(benchmark):
    """Benchmark the parse_generic_llm_error function when ApiResponseException is passed."""
    error = ApiResponseException(
        response={
            "error": "this is error",
            "message": "this is error message",
            "extensions": {},
            "status_code": 400,
        },
        message=None,
    )

    # benchmark the function to parse LLM errors
    benchmark(errors_parsing.parse_generic_llm_error, error)


def test_parse_generic_llm_error_on_api_response_exception_with_message(benchmark):
    """Benchmark the parse_generic_llm_error function when ApiResponseException is passed."""
    message = "This is proper error message"

    error = ApiResponseException(
        message=message,
        response={
            "message": "this is error message",
            "error": "this is error",
            "extensions": {},
            "status_code": 400,
        },
    )

    # benchmark the function to parse LLM errors
    benchmark(errors_parsing.parse_generic_llm_error, error)


def test_parse_generic_llm_error_on_api_response_exception_without_extensions_state(
    benchmark,
):
    """Test the parse_generic_llm_error function when ApiResponseException is passed."""
    message = "This is proper error message"

    error = ApiResponseException(
        message=message,
        response={
            "message": "this is error message",
            "error": "this is error",
            "extensions": {},
            "status_code": 400,
        },
    )
    # cleanup extensions state
    error.response.extensions.state = None

    # try to parse the exception
    errors_parsing.parse_generic_llm_error(error)

    # benchmark the function to parse LLM errors
    benchmark(errors_parsing.parse_generic_llm_error, error)


def test_parse_generic_llm_error_on_api_response_exception_with_extensions_state(
    benchmark,
):
    """Benchmark the parse_generic_llm_error function when ApiResponseException is passed."""
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
            "status_code": 400,
        },
    )
    # cleanup extensions state
    error.response.extensions.state = None

    # benchmark the function to parse LLM errors
    benchmark(errors_parsing.parse_generic_llm_error, error)


def test_parse_generic_llm_error_on_unknown_exception(benchmark):
    """Benchmark the parse_generic_llm_error function when unknown exception is passed."""
    # generic exception
    error = Exception("Exception")

    # benchmark the function to parse LLM errors
    benchmark(errors_parsing.parse_generic_llm_error, error)


def test_parse_generic_llm_error_on_unknown_exception_without_message(benchmark):
    """Benchmark the parse_generic_llm_error function when unknown exception is passed."""
    # generic exception
    error = Exception()

    # benchmark the function to parse LLM errors
    benchmark(errors_parsing.parse_generic_llm_error, error)
