"""Utility functions for LLM providers configuration."""

import json


def credentials_str_to_dict(credentials_json: str) -> dict[str, str]:
    """Parse a single-level JSON object into string keys and string values."""
    parsed = json.loads(credentials_json)
    if not isinstance(parsed, dict):
        msg = "credentials must be a JSON object with string values"
        raise TypeError(msg)
    return {str(k): str(v) for k, v in parsed.items()}
