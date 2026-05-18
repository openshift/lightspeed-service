"""Module for parsing errors from different LLMs."""

import json
import logging

import httpx
from fastapi import status
from ibm_watsonx_ai.wml_client_error import ApiRequestFailure
from openai import APIConnectionError, APITimeoutError, BadRequestError

from ols import config

logger = logging.getLogger(__name__)

# Constants for default messages and status codes
DEFAULT_ERROR_MESSAGE = "An error occurred during LLM invocation. Please contact your OpenShift Lightspeed administrator."  # noqa: E501
DEFAULT_STATUS_CODE = status.HTTP_500_INTERNAL_SERVER_ERROR

_NETWORK_PREFIX = "[Network]"
_LLM_BACKEND_PREFIX = "[LLM Backend]"

NETWORK_ERROR_MSG = (
    "Unable to reach the LLM backend. "
    "Please verify network connectivity and provider URL configuration."
)

PROMPT_TOO_LONG_WITH_TOOLS_ERROR_MSG = (
    "The request exceeds the model's token limit, possibly due to tool "
    "usage. This usually means the configured context_window_size is larger "
    "than what the model actually supports. Please verify that "
    "context_window_size matches the model's real context window, or "
    "refine your prompt to be more specific."
)

PROMPT_TOO_LONG_ERROR_MSG = (
    "The prompt exceeds the model's token limit. This usually means "
    "the configured context_window_size is larger than what the model "
    "actually supports. Please verify that context_window_size in the "
    "model configuration matches the model's real context window."
)


def parse_openai_error(e: BadRequestError) -> tuple[int, str, str]:
    """Parse OpenAI or Azure error."""
    if e.body is not None and isinstance(e.body, dict) and "message" in e.body:
        response_text = e.body["message"]
    else:
        response_text = e.message
    return e.status_code, response_text, e.message


def parse_watsonx_error(e: ApiRequestFailure) -> tuple[int, str, str]:
    """Parse Watsonx error."""
    status_code = e.response.status_code
    logger.error(
        "WatsonX API error: status_code=%d, reason=%s, url=%s, error_msg=%s, "
        "response_text=%s",
        status_code,
        e.response.reason,
        getattr(e.response, "url", "unknown"),
        e.error_msg,
        e.response.text[:500] if e.response.text else "empty",
    )
    try:
        errors = json.loads(e.response.text)["errors"]
        if len(errors) != 1 or errors[0].get("message") is None:
            raise ValueError
        response_text = errors[0]["message"]
    except (json.JSONDecodeError, KeyError, ValueError):
        response_text = e.response.reason
    return status_code, response_text, e.error_msg


def parse_generic_llm_error(e: Exception) -> tuple[int, str, str]:
    """Parse LLM error and prefix the response with error source."""
    logger.error(
        "LLM error received: type=%s, message=%s",
        type(e).__name__,
        str(e)[:500],
    )
    match e:
        case APIConnectionError() | APITimeoutError():
            return (
                status.HTTP_502_BAD_GATEWAY,
                f"{_NETWORK_PREFIX} {NETWORK_ERROR_MSG}",
                str(e),
            )
        case httpx.ConnectError() | httpx.ConnectTimeout():
            return (
                status.HTTP_502_BAD_GATEWAY,
                f"{_NETWORK_PREFIX} {NETWORK_ERROR_MSG}",
                str(e),
            )
        case BadRequestError():
            sc, resp, cause = parse_openai_error(e)
            return sc, f"{_LLM_BACKEND_PREFIX} {resp}", cause
        case ApiRequestFailure():
            sc, resp, cause = parse_watsonx_error(e)
            return sc, f"{_LLM_BACKEND_PREFIX} {resp}", cause
        case _:
            return (
                DEFAULT_STATUS_CODE,
                f"{_LLM_BACKEND_PREFIX} {DEFAULT_ERROR_MESSAGE}",
                str(e),
            )


def handle_known_errors(response: str, cause: str) -> tuple[str, str]:
    """Handle known errors and return a user-friendly message."""
    if all(
        [
            "maximum" in response.lower(),
            "context" in response.lower(),
            "length" in response.lower(),
        ]
    ):
        if config.mcp_servers.servers:
            return (
                f"{_LLM_BACKEND_PREFIX} {PROMPT_TOO_LONG_WITH_TOOLS_ERROR_MSG}",
                cause,
            )
        return f"{_LLM_BACKEND_PREFIX} {PROMPT_TOO_LONG_ERROR_MSG}", cause

    return response, cause
