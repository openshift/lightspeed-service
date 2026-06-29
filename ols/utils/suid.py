"""Session ID utility functions."""

import uuid


def get_suid() -> str:
    """Generate a unique session ID (SUID) using UUID4.

    Returns:
        A unique session ID.
    """
    return str(uuid.uuid4())


def conversation_id_to_trace_id(conversation_id: str) -> str:
    """Convert a UUID conversation_id to a 32-hex-char OTEL trace ID."""
    return conversation_id.replace("-", "")


def check_suid(suid: str) -> bool:
    """Check if given string is a proper session ID."""
    try:
        # accepts strings and bytes only
        uuid.UUID(suid)
        return True
    except (ValueError, TypeError):
        return False
