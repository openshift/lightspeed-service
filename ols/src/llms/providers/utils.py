"""Utility functions for LLM providers configuration."""

import json

# Vertex AI service account requires this scope to access the API
# https://github.com/googleapis/python-genai/issues/2#issuecomment-2537279484
VERTEX_AI_OAUTH_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/cloud-platform",
)


def credentials_str_to_dict(credentials_json: str) -> dict[str, str]:
    """Parse a single-level JSON object into string keys and string values."""
    parsed = json.loads(credentials_json)
    if not isinstance(parsed, dict):
        msg = "credentials must be a JSON object with string values"
        raise TypeError(msg)
    return {str(k): str(v) for k, v in parsed.items()}
