"""This module contains the FastAPI endpoint for the OLS streaming query endpoint."""

import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ols import config, constants
from ols.app.endpoints.ols import (
    generate_response,
    log_processing_durations,
    process_request,
    store_conversation_history,
    store_transcript,
)
from ols.app.models.models import (
    Attachment,
    LLMRequest,
    SummarizerResponse,
)
from ols.constants import MEDIA_TYPE_TEXT
from ols.src.auth.auth import get_auth_dependency

logger = logging.getLogger(__name__)

router = APIRouter(tags=["streaming_query"])
auth_dependency = get_auth_dependency(config.ols_config, virtual_path="/ols-access")


@router.post("/streaming_query")
def conversation_request(llm_request: LLMRequest, auth: Any = Depends(auth_dependency)):
    """Handle conversation requests for the OLS endpoint.

    Args:
        llm_request: The request containing a query, conversation ID, and optional attachments.
        auth: The Authentication handler (FastAPI Depends) that will handle authentication Logic.

    Returns:
        Response containing the processed information.
    """
    (
        user_id,
        conversation_id,
        query_without_attachments,
        previous_input,
        attachments,
        valid,
        timestamps,
    ) = process_request(auth, llm_request)

    if not valid:

        async def invalid_response_generator():
            yield constants.INVALID_QUERY_RESP

        summarizer_response = invalid_response_generator()
    else:
        summarizer_response = generate_response(
            conversation_id, llm_request, previous_input, streaming=True
        )

    return StreamingResponse(
        response_processing_wrapper(
            summarizer_response,
            user_id,
            conversation_id,
            llm_request,
            attachments,
            valid,
            query_without_attachments,
            llm_request.media_type,
            timestamps,
        ),
        media_type=llm_request.media_type,
    )


async def response_processing_wrapper(
    generator: Any,
    user_id: str,
    conversation_id: str,
    llm_request: LLMRequest,
    attachments: list[Attachment],
    valid: bool,
    query_without_attachments: str,
    media_type: str,
    timestamps: dict[str, float],
):
    """Wrap the processing of the response to include the referenced documents."""
    response = ""
    rag_chunks = []
    history_truncated = False
    idx = 0
    async for item in generator:
        # drain the stream until we get the SummarizerResponse (end of
        # LLM response) and yield the doc links at the end
        if isinstance(item, SummarizerResponse):
            history_truncated = item.history_truncated
            rag_chunks = item.rag_chunks
            break

        response += item
        if media_type == MEDIA_TYPE_TEXT:
            yield item
        else:
            # if it is not text, it is JSON - there are no more media
            # types now
            yield json.dumps({"event": "token", "data": {"id": idx, "token": item}})
        idx += 1

    timestamps["generate response"] = time.time()

    store_conversation_history(
        user_id, conversation_id, llm_request, response, attachments
    )

    if config.ols_config.user_data_collection.transcripts_disabled:
        logger.debug("transcripts collections is disabled in configuration")
    else:
        store_transcript(
            user_id,
            conversation_id,
            valid,
            query_without_attachments,
            llm_request,
            response,
            rag_chunks,
            history_truncated,
            attachments,
        )

    timestamps["store transcripts"] = time.time()

    ref_docs_string = "\n".join(
        f"{title}: {url}"
        for title, url in {
            rag_chunk.doc_title: rag_chunk.doc_url for rag_chunk in rag_chunks
        }.items()
    )

    timestamps["add references"] = time.time()
    log_processing_durations(timestamps)

    if media_type == MEDIA_TYPE_TEXT:
        if ref_docs_string != "":
            yield f"\n\n---\n\n{ref_docs_string}"
    else:
        # if it is not text, it is JSON - there are no more media
        # types now
        for rag_chunk in rag_chunks:
            yield json.dumps(
                {
                    "event": "doc_references",
                    "data": {
                        "doc_title": rag_chunk.doc_title,
                        "doc_url": rag_chunk.doc_url,
                    },
                }
            )
