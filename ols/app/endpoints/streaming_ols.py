"""This module contains the FastAPI endpoint for the OLS streaming query endpoint."""

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ols import config, constants
from ols.app.endpoints.ols import (
    generate_response,
    log_processing_durations,
    redact_attachments,
    redact_query,
    retrieve_attachments,
    retrieve_conversation_id,
    retrieve_previous_input,
    retrieve_user_id,
    store_conversation_history,
    store_transcript,
    validate_question,
    validate_requested_provider_model,
)
from ols.app.models.models import (
    LLMRequest,
    SummarizerResponse,
)
from ols.src.auth.auth import get_auth_dependency
from ols.src.query_helpers.attachment_appender import append_attachments_to_query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["streaming_query"])
auth_dependency = get_auth_dependency(config.ols_config, virtual_path="/ols-access")


# TODO: more unification of the streaming and non-streaming endpoints,
# half of the endpoint is practically the same
# TODO: correctly mark the endpoint return type - consider using JSONlines
# instead of streaming plain text
@router.post("/streaming_query")
def conversation_request(llm_request: LLMRequest, auth: Any = Depends(auth_dependency)):
    """Handle conversation requests for the OLS endpoint.

    Args:
        llm_request: The request containing a query, conversation ID, and optional attachments.
        auth: The Authentication handler (FastAPI Depends) that will handle authentication Logic.

    Returns:
        Response containing the processed information.
    """
    timestamps: dict[str, float] = {}
    timestamps["start"] = time.time()

    # Initialize variables
    previous_input = []

    user_id = retrieve_user_id(auth)
    logger.info("User ID %s", user_id)
    timestamps["retrieve user"] = time.time()

    conversation_id = retrieve_conversation_id(llm_request)
    timestamps["retrieve conversation"] = time.time()

    # Important note: Redact the query before attempting to do any
    # logging of the query to avoid leaking PII into logs.

    # Redact the query
    llm_request = redact_query(conversation_id, llm_request)
    timestamps["redact query"] = time.time()

    # Log incoming request (after redaction)
    logger.info("%s Incoming request: %s", conversation_id, llm_request.query)

    previous_input = retrieve_previous_input(user_id, llm_request)
    timestamps["retrieve previous input"] = time.time()

    # Retrieve attachments from the request
    attachments = retrieve_attachments(llm_request)

    # Redact all attachments
    attachments = redact_attachments(conversation_id, attachments)

    # All attachments should be appended to query - but store original
    # query for later use in transcript storage
    query_without_attachments = llm_request.query
    llm_request.query = append_attachments_to_query(llm_request.query, attachments)
    timestamps["append attachments"] = time.time()

    validate_requested_provider_model(llm_request)

    # Validate the query
    if not previous_input:
        valid = validate_question(conversation_id, llm_request)
    else:
        logger.debug("follow-up conversation - skipping question validation")
        valid = True

    timestamps["validate question"] = time.time()

    if not valid:
        summarizer_response = SummarizerResponse(
            constants.INVALID_QUERY_RESP,
            [],
            False,
        )
    else:
        summarizer_response = generate_response(
            conversation_id, llm_request, previous_input, streaming=True
        )

    return StreamingResponse(
        response_processing_wrapper(
            summarizer_response,
            timestamps,
            user_id,
            conversation_id,
            llm_request,
            attachments,
            valid,
            query_without_attachments,
        ),
        media_type="text/plain",
    )


async def response_processing_wrapper(
    generator,
    timestamps,
    user_id,
    conversation_id,
    llm_request,
    attachments,
    valid,
    query_without_attachments,
):
    """Wrap the processing of the response to include the referenced documents."""
    # drain the stream until we get the SummarizerResponse (end of LLM
    # response) and yield the doc links at the end
    response = ""
    async for item in generator:
        if isinstance(item, SummarizerResponse):
            history_truncated = item.history_truncated
            rag_chunks = item.rag_chunks
            break
        response += item
        yield item

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

    yield f"\n\n---\n\n{ref_docs_string}"
