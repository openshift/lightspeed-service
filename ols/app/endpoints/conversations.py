"""Handlers for conversations management REST API endpoints."""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from ols import config
from ols.app.models.models import (
    ConversationDeleteResponse,
    ConversationDetailResponse,
    ConversationsListResponse,
    ConversationUpdateRequest,
    ConversationUpdateResponse,
    ErrorResponse,
    ForbiddenResponse,
    UnauthorizedResponse,
)
from ols.src.auth.auth import get_auth_dependency
from ols.utils import suid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])
auth_dependency = get_auth_dependency(config.ols_config, virtual_path="/ols-access")


def retrieve_user_id(auth: Any) -> str:
    """Retrieve user ID from the token processed by auth. mechanism."""
    return auth[0]


def retrieve_skip_user_id_check(auth: Any) -> bool:
    """Retrieve skip user_id check from the token processed by auth. mechanism."""
    return auth[2]


list_conversations_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "List of conversations retrieved successfully",
        "model": ConversationsListResponse,
    },
    401: {
        "description": "Missing or invalid credentials provided by client",
        "model": UnauthorizedResponse,
    },
    403: {
        "description": "Client does not have permission to access resource",
        "model": ForbiddenResponse,
    },
    500: {
        "description": "Internal server error",
        "model": ErrorResponse,
    },
}


@router.get("", responses=list_conversations_responses)
def list_conversations(
    auth: Annotated[Any, Depends(auth_dependency)],
) -> ConversationsListResponse:
    """List all conversations for the authenticated user.

    Args:
        auth: The Authentication handler (FastAPI Depends) that will
            handle authentication Logic.

    Returns:
        Response containing list of conversations with metadata.
    """
    user_id = retrieve_user_id(auth)
    skip_user_id_check = retrieve_skip_user_id_check(auth)

    logger.debug("Listing conversations for user %s", user_id)

    try:
        conversations = config.conversation_cache.list(user_id, skip_user_id_check)
        return ConversationsListResponse(conversations=conversations)
    except Exception as e:
        logger.error("Error listing conversations: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error listing conversations",
                "cause": str(e),
            },
        ) from e


get_conversation_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Conversation retrieved successfully",
        "model": ConversationDetailResponse,
    },
    400: {
        "description": "Invalid conversation ID format",
        "model": ErrorResponse,
    },
    401: {
        "description": "Missing or invalid credentials provided by client",
        "model": UnauthorizedResponse,
    },
    403: {
        "description": "Client does not have permission to access resource",
        "model": ForbiddenResponse,
    },
    404: {
        "description": "Conversation not found",
        "model": ErrorResponse,
    },
    500: {
        "description": "Internal server error",
        "model": ErrorResponse,
    },
}


@router.get("/{conversation_id}", responses=get_conversation_responses)
def get_conversation(
    conversation_id: str,
    auth: Annotated[Any, Depends(auth_dependency)],
) -> ConversationDetailResponse:
    """Get a specific conversation by ID.

    Args:
        conversation_id: The conversation ID to retrieve.
        auth: The Authentication handler (FastAPI Depends) that will
            handle authentication Logic.

    Returns:
        Response containing conversation details with chat history.
    """
    user_id = retrieve_user_id(auth)
    skip_user_id_check = retrieve_skip_user_id_check(auth)

    if not suid.check_suid(conversation_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "response": "Invalid conversation ID format",
                "cause": f"Conversation ID {conversation_id} is not valid",
            },
        )

    logger.debug("Getting conversation %s for user %s", conversation_id, user_id)

    try:
        cache_entries = config.conversation_cache.get(
            user_id, conversation_id, skip_user_id_check
        )

        if not cache_entries:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "response": "Conversation not found",
                    "cause": f"Conversation {conversation_id} does not exist",
                },
            )

        # Transform cache entries to chat history format
        chat_history = []
        for entry in cache_entries:
            messages = [
                {"type": "user", "content": entry.query.content},
                {"type": "assistant", "content": entry.response.content},
            ]
            chat_history.append(
                {
                    "messages": messages,
                    "tool_calls": entry.tool_calls,
                    "tool_results": entry.tool_results,
                }
            )

        return ConversationDetailResponse(
            conversation_id=conversation_id,
            chat_history=chat_history,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting conversation %s: %s", conversation_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error getting conversation",
                "cause": str(e),
            },
        ) from e


delete_conversation_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Conversation deleted successfully",
        "model": ConversationDeleteResponse,
    },
    400: {
        "description": "Invalid conversation ID format",
        "model": ErrorResponse,
    },
    401: {
        "description": "Missing or invalid credentials provided by client",
        "model": UnauthorizedResponse,
    },
    403: {
        "description": "Client does not have permission to access resource",
        "model": ForbiddenResponse,
    },
    500: {
        "description": "Internal server error",
        "model": ErrorResponse,
    },
}


@router.delete("/{conversation_id}", responses=delete_conversation_responses)
def delete_conversation(
    conversation_id: str,
    auth: Annotated[Any, Depends(auth_dependency)],
) -> ConversationDeleteResponse:
    """Delete a specific conversation by ID.

    Args:
        conversation_id: The conversation ID from the URL path.
        auth: The Authentication handler (FastAPI Depends) that will
            handle authentication Logic.

    Returns:
        Response indicating whether the deletion was successful.
    """
    user_id = retrieve_user_id(auth)
    skip_user_id_check = retrieve_skip_user_id_check(auth)

    if not suid.check_suid(conversation_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "response": "Invalid conversation ID format",
                "cause": f"Conversation ID {conversation_id} is not valid",
            },
        )

    logger.debug("Deleting conversation %s for user %s", conversation_id, user_id)

    try:
        deleted = config.conversation_cache.delete(
            user_id, conversation_id, skip_user_id_check
        )

        if deleted:
            return ConversationDeleteResponse(
                conversation_id=conversation_id,
                response="Conversation deleted successfully",
                success=True,
            )
        return ConversationDeleteResponse(
            conversation_id=conversation_id,
            response="Conversation not found",
            success=False,
        )
    except Exception as e:
        logger.error("Error deleting conversation %s: %s", conversation_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error deleting conversation",
                "cause": str(e),
            },
        ) from e


update_conversation_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Conversation updated successfully",
        "model": ConversationUpdateResponse,
    },
    400: {
        "description": "Invalid conversation ID format",
        "model": ErrorResponse,
    },
    401: {
        "description": "Missing or invalid credentials provided by client",
        "model": UnauthorizedResponse,
    },
    403: {
        "description": "Client does not have permission to access resource",
        "model": ForbiddenResponse,
    },
    404: {
        "description": "Conversation not found",
        "model": ErrorResponse,
    },
    500: {
        "description": "Internal server error",
        "model": ErrorResponse,
    },
}


@router.put("/{conversation_id}", responses=update_conversation_responses)
def update_conversation(
    conversation_id: str,
    update_request: ConversationUpdateRequest,
    auth: Annotated[Any, Depends(auth_dependency)],
) -> ConversationUpdateResponse:
    """Update a conversation's metadata (topic_summary).

    Args:
        conversation_id: The conversation ID from the URL path.
        update_request: The request body containing topic_summary.
        auth: The Authentication handler (FastAPI Depends) that will
            handle authentication Logic.

    Returns:
        Response indicating whether the update was successful.
    """
    user_id = retrieve_user_id(auth)
    skip_user_id_check = retrieve_skip_user_id_check(auth)

    if not suid.check_suid(conversation_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "response": "Invalid conversation ID format",
                "cause": f"Conversation ID {conversation_id} is not valid",
            },
        )

    logger.debug(
        "Updating conversation %s topic_summary for user %s", conversation_id, user_id
    )

    try:
        cache_entries = config.conversation_cache.get(
            user_id, conversation_id, skip_user_id_check
        )

        if not cache_entries:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "response": "Conversation not found",
                    "cause": f"Conversation {conversation_id} does not exist",
                },
            )

        config.conversation_cache.set_topic_summary(
            user_id, conversation_id, update_request.topic_summary, skip_user_id_check
        )

        return ConversationUpdateResponse(
            conversation_id=conversation_id,
            success=True,
            message="Topic summary updated successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating conversation %s: %s", conversation_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error updating conversation",
                "cause": str(e),
            },
        ) from e
