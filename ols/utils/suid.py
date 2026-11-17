"""Session ID utility functions."""

import uuid

# Deterministic namespace for opaque Kubernetes/TokenReview user.uid values
# (for example Entra JWT `sub` on HCP external authentication).
_USER_UID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://redhat.com/openshift-lightspeed/user-uid/v1"
)


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


def normalize_user_uid(uid: str) -> str:
    """Map a Kubernetes user UID to a value safe for conversation-cache keys.

    TokenReview often returns opaque OIDC subjects that are not RFC-4122 UUIDs.
    Conversation history requires UUID-shaped user IDs. UUID values are returned
    unchanged; other strings are mapped with UUID5 for stable per-user keys.

    Args:
        uid: User UID from Kubernetes TokenReview.

    Returns:
        A UUID string suitable for conversation-cache keys.
    """
    if check_suid(uid):
        return uid
    return str(uuid.uuid5(_USER_UID_NAMESPACE, uid))


def normalize_k8s_user_uid(uid: str | None, username: str) -> str:
    """Map TokenReview identity to a UUID-shaped user id for cache and API use.

    Args:
        uid: User UID from TokenReview, if present.
        username: Authenticated Kubernetes username from TokenReview.

    Returns:
        A UUID string suitable for conversation-cache keys.
    """
    if uid:
        return normalize_user_uid(uid)
    return str(uuid.uuid5(_USER_UID_NAMESPACE, f"username:{username}"))
