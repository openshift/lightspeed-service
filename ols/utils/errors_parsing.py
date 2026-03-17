"""Module for parsing errors from different LLMs."""

import json
import logging

from fastapi import status
from ibm_watsonx_ai.wml_client_error import ApiRequestFailure
from openai import BadRequestError

from ols import config

logger = logging.getLogger(__name__)

# Constants for default messages and status codes
DEFAULT_ERROR_MESSAGE = "An error occurred during LLM invocation. Please contact your OpenShift Lightspeed administrator."  # noqa: E501
DEFAULT_STATUS_CODE = status.HTTP_500_INTERNAL_SERVER_ERROR

PROMPT_TOO_LONG_WITH_TOOLS_ERROR_MSG = (
    "The request exceeds the allowed token limit, possibly due to tool "
    "usage or misconfigured context window or maximum response tokens "
    "settings. Please refine your prompt to be more specific, or verify "
    "that the token-related configuration parameters are correctly set."
)

# The msg is user facing - note the config fields from OLS CRD
PROMPT_TOO_LONG_ERROR_MSG = (
    "The prompt exceeds the maximum token limit. Please shorten your "
    "input or adjust the context window and maximum response tokens "
    "settings in the configuration to allow for larger prompts."
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
        # fallback to response reason if message is not found
        response_text = e.response.reason
    return status_code, response_text, e.error_msg


def parse_generic_llm_error(e: Exception) -> tuple[int, str, str]:
    """Try to parse generic LLM error."""
    logger.error(
        "LLM error received: type=%s, message=%s",
        type(e).__name__,
        str(e)[:500],
    )
    match e:
        case BadRequestError():
            return parse_openai_error(e)
        case ApiRequestFailure():
            return parse_watsonx_error(e)
        case _:
            return DEFAULT_STATUS_CODE, DEFAULT_ERROR_MESSAGE, str(e)


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
            # tool calls are too long
            return PROMPT_TOO_LONG_WITH_TOOLS_ERROR_MSG, cause
        # models with context smaller than our default
        return PROMPT_TOO_LONG_ERROR_MSG, cause

    return response, cause
