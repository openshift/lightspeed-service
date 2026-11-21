"""Conversation access checks for client-supplied conversation IDs."""

import logging

from fastapi import HTTPException, status

from ols import config

logger = logging.getLogger(__name__)


def ensure_conversation_owned_by_user(user_id: str, conversation_id: str) -> None:
    """Reject requests that target a conversation owned by another user.

    Client-provided conversation IDs must not grant access to another user's
    cached history. Under normal k8s auth each principal has a distinct user_id,
    so cache lookup would already miss; this check blocks ID reuse and makes the
    failure explicit when a conversation_id is already bound to a different owner.
    """
    owner_user_id = config.conversation_cache.find_owner_user_id(conversation_id)
    if owner_user_id is None or owner_user_id == user_id:
        return

    logger.warning(
        "Conversation %s owned by different user (requester=%s)",
        conversation_id,
        user_id,
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "response": "Forbidden",
            "cause": "Conversation is not owned by the authenticated user",
        },
    )