"""Benchmarks for errors parsing feature."""

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
