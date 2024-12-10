"""FastAPI endpoint for the OLS streaming query."""

import json
import logging
import time
from typing import Any, AsyncGenerator

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
from ols.app.models.models import Attachment, LLMRequest, SummarizerResponse
from ols.constants import MEDIA_TYPE_TEXT
from ols.src.auth.auth import get_auth_dependency
from ols.utils import errors_parsing
from ols.utils.token_handler import PromptTooLongError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["streaming_query"])
auth_dependency = get_auth_dependency(config.ols_config, virtual_path="/ols-access")


@router.post("/streaming_query")
def conversation_request(llm_request: LLMRequest, auth: Any = Depends(auth_dependency)):
    """Handle conversation requests for the OLS endpoint."""
    (
        user_id,
        conversation_id,
        query_without_attachments,
        previous_input,
        attachments,
        valid,
        timestamps,
    ) = process_request(auth, llm_request)

    summarizer_response = (
        invalid_response_generator()
        if not valid
        else generate_response(
            conversation_id, llm_request, previous_input, streaming=True
        )
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


async def invalid_response_generator() -> AsyncGenerator[str, None]:
    """Yield an invalid query response."""
    yield constants.INVALID_QUERY_RESP


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
    """Process the response from the generator and handle metadata and errors."""
    response = ""
    rag_chunks = []
    history_truncated = False
    idx = 0
    try:
        async for item in generator:
            if isinstance(item, SummarizerResponse):
                rag_chunks = item.rag_chunks
                history_truncated = item.history_truncated
                break

            response += item
            yield build_yield_item(item, idx, media_type)
            idx += 1
    # NOTE: These errors happen inside of the actual generator - in
    # the StreamingResponse - once we are in, we can only return bytes.
    # So we can't return status code here, we just yield the error
    # message (format depends on the media type).
    except PromptTooLongError as summarizer_error:
        yield prompt_too_long_error(summarizer_error, media_type)
    except Exception as summarizer_error:
        yield generic_llm_error(summarizer_error, media_type)
    timestamps["generate response"] = time.time()

    store_data(
        user_id,
        conversation_id,
        llm_request,
        response,
        attachments,
        valid,
        query_without_attachments,
        rag_chunks,
        history_truncated,
        timestamps,
    )

    async for doc_reference in yield_references(rag_chunks, media_type):
        yield doc_reference
    timestamps["add references"] = time.time()

    log_processing_durations(timestamps)


def build_yield_item(item: str, idx: int, media_type: str) -> str:
    """Build an item to yield based on media type."""
    if media_type == MEDIA_TYPE_TEXT:
        return item
    return json.dumps({"event": "token", "data": {"id": idx, "token": item}})


def prompt_too_long_error(error: PromptTooLongError, media_type: str):
    """Raise an HTTP exception for long prompts."""
    logger.error("Prompt is too long: %s", error)
    if media_type == MEDIA_TYPE_TEXT:
        return f"Prompt is too long: {error}"
    return json.dumps(
        {
            "event": "error",
            "data": {
                "response": "Prompt is too long",
                "cause": str(error),
            },
        }
    )


def generic_llm_error(error: Exception, media_type: str):
    """Raise an HTTP exception for generic LLM errors."""
    logger.error("Error while obtaining answer for user question")
    logger.exception(error)
    _, response, cause = errors_parsing.parse_generic_llm_error(error)

    if media_type == MEDIA_TYPE_TEXT:
        return f"{response}: {cause}"
    return json.dumps(
        {
            "event": "error",
            "data": {
                "response": response,
                "cause": cause,
            },
        }
    )


def store_data(
    user_id: str,
    conversation_id: str,
    llm_request: LLMRequest,
    response: str,
    attachments: list[Attachment],
    valid: bool,
    query_without_attachments: str,
    rag_chunks: list[Attachment],
    history_truncated: bool,
    timestamps: dict[str, float],
):
    """Store conversation history and transcript if enabled."""
    store_conversation_history(
        user_id, conversation_id, llm_request, response, attachments
    )

    if not config.ols_config.user_data_collection.transcripts_disabled:
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


async def yield_references(rag_chunks: list[Attachment], media_type: str):
    """Yield document references based on media type."""
    if media_type == MEDIA_TYPE_TEXT:
        ref_docs_string = "\n".join(
            f"{title}: {url}"
            for title, url in {
                rag_chunk.doc_title: rag_chunk.doc_url for rag_chunk in rag_chunks
            }.items()
        )
        if ref_docs_string:
            yield f"\n\n---\n\n{ref_docs_string}"
    else:
        for chunk in rag_chunks:
            yield json.dumps(
                {
                    "event": "doc_references",
                    "data": {
                        "doc_title": chunk.doc_title,
                        "doc_url": chunk.doc_url,
                    },
                }
            )
