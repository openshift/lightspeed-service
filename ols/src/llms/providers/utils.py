"""Utility functions for LLM providers configuration."""

import json
from typing import Any

from google.auth.credentials import Credentials as GoogleCredentials
from google.oauth2 import credentials as oauth2_credentials
from google.oauth2 import service_account

from ols.app.models.config import ModelParameters

# Vertex AI service account requires this scope to access the API
# https://github.com/googleapis/python-genai/issues/2#issuecomment-2537279484
VERTEX_AI_OAUTH_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/cloud-platform",
)


def load_vertex_credentials(credentials_json: str) -> GoogleCredentials:
    """Build Vertex-scoped Google credentials from JSON file contents.

    Supports ``type: service_account`` (key file) and ``type: authorized_user``
    (refresh-token / user OAuth JSON as produced by application-default login).
    The JSON object must include a non-empty ``type`` field.

    Args:
        credentials_json: Raw JSON string read from the credentials file.

    Returns:
        Credentials instance scoped with ``VERTEX_AI_OAUTH_SCOPES``.

    Raises:
        TypeError: If JSON does not decode to an object.
        ValueError: If type is missing/unsupported or required fields are absent.
    """
    parsed = json.loads(credentials_json)
    if not isinstance(parsed, dict):
        msg = "credentials must be a JSON object"
        raise TypeError(msg)
    cred_type = parsed.get("type")
    if not cred_type:
        msg = 'Google credentials JSON must include a non-empty string "type" field'
        raise ValueError(msg)
    scopes = list(VERTEX_AI_OAUTH_SCOPES)
    if cred_type == "service_account":
        return service_account.Credentials.from_service_account_info(
            parsed,
            scopes=scopes,
        )  # type: ignore[no-untyped-call]
    if cred_type == "authorized_user":
        return oauth2_credentials.Credentials.from_authorized_user_info(
            parsed,
            scopes=scopes,
        )  # type: ignore[no-untyped-call]
    msg = f"Unsupported Google credential type for Vertex: {cred_type!r}"
    raise ValueError(msg)


def anthropic_model_profile_max_tokens(model: str) -> int:
    """Return the Anthropic model-profile output limit for a model name.

    The Vertex and Bedrock adapters do not fill an unset output limit from the
    model profile: the Vertex adapter hard-defaults to 4096 and the Bedrock
    client applies its own low cap. A hardcoded cap below the profile default
    silently truncates single-response structured output, because reasoning
    tokens count against the same limit. Apply the model-profile default that
    the direct Anthropic adapter would use, falling back to its fallback cap
    for unknown models.

    Args:
        model: Model name or Bedrock inference-profile ID (for example
            ``global.anthropic.claude-sonnet-5-5-v1:0``). Profiles are keyed
            by bare model names.

    Returns:
        The model-profile maximum output tokens.
    """
    from langchain_anthropic.chat_models import (  # pylint: disable=import-outside-toplevel
        _FALLBACK_MAX_OUTPUT_TOKENS,
        _get_default_model_profile,
    )

    bare_model = model.split(":")[0].rsplit(".", 1)[-1]
    profile = _get_default_model_profile(bare_model) or {}
    return int(profile.get("max_output_tokens", _FALLBACK_MAX_OUTPUT_TOKENS))


def populate_openai_reasoning(
    params: ModelParameters, default_parameters: dict[str, Any]
) -> None:
    """Populate reasoning configuration from model parameters."""
    reasoning_config = params.reasoning_config or {}

    if reasoning_config:
        reasoning_parameters = dict(reasoning_config)
        if "verbosity" in reasoning_parameters:
            default_parameters["verbosity"] = reasoning_parameters.pop("verbosity")
        default_parameters["reasoning"] = reasoning_parameters


def populate_vllm_reasoning(
    params: ModelParameters, default_parameters: dict[str, Any]
) -> None:
    """Populate reasoning configuration for vLLM (Chat Completions API).

    Maps reasoning_config to vLLM-compatible parameters instead of using
    the Responses API (which is not supported by vLLM). Maps effort setting
    to reasoning_effort and ensures use_responses_api is False.
    """
    default_parameters["use_responses_api"] = False
    reasoning_config = params.reasoning_config or {}

    if not reasoning_config:
        return

    reasoning_parameters = dict(reasoning_config)

    # Map effort to reasoning_effort for vLLM
    if "effort" in reasoning_parameters:
        effort = reasoning_parameters.pop("effort")
        default_parameters["reasoning_effort"] = effort

    # Preserve other parameters (summary, verbosity, etc.)
    if reasoning_parameters:
        default_parameters["reasoning"] = reasoning_parameters
