"""Session ID utility functions."""

import uuid


def get_suid() -> str:
    """Generate a unique session ID (SUID) using UUID4.

    Returns:
        A unique session ID.
    """
    return str(uuid.uuid4())


def check_suid(suid: str) -> bool:
    """Check if given string is a proper session ID."""
    try:
        # accepts strings and bytes only
        uuid.UUID(suid)
        return True
    except (ValueError, TypeError):
        return False


# maximal length of an externally-provided user ID
USER_ID_MAX_LENGTH = 256


def check_user_id(user_id: str) -> bool:
    """Check if given string can be used as a user ID.

    Unlike session (conversation) IDs, which this service generates itself,
    user IDs come from the authentication layer. For example the Kubernetes
    `k8s` authentication module uses the `uid` field of a TokenReview
    response, whose format is controlled by the authenticator and is not
    guaranteed to be a UUID: the Kubernetes API server maps it from an OIDC
    claim (commonly `sub`), so it can be an issuer-specific opaque string
    such as a snowflake ID (`305070630980550817`) or `sha256~<hash>`.

    A user ID is therefore checked as a bounded opaque string:
    - must be a non-empty string of at most `USER_ID_MAX_LENGTH` characters,
    - must not contain whitespace or other non-printable characters,
    - must not contain `:`, the compound-key separator used to combine the
      user ID and the (UUID-formatted) conversation ID into one cache key.
      Allowing `:` would let a user whose ID is itself a `"<other>:..."`
      prefix enumerate another user's conversation keys when listing.
      Note that this keeps the already-existing behavior for Kubernetes
      service account UIDs (`system:serviceaccount:<ns>:<sa>`), which are
      rejected by the previous UUID check too.

    Args:
        user_id: The user ID to check.

    Returns:
        True if the user ID can be used, False otherwise.
    """
    if not isinstance(user_id, str):
        return False
    if not 1 <= len(user_id) <= USER_ID_MAX_LENGTH:
        return False
    if ":" in user_id:
        return False
    # str.isprintable() is False for control and other non-printable
    # characters; it does not cover whitespace, so check isspace() too
    # (the ASCII space character is considered "printable").
    return all(not c.isspace() and c.isprintable() for c in user_id)
