"""FastAPI endpoint for the OLS streaming query.

This module defines the endpoint and supporting functions for handling
streaming queries.
"""

import json
import logging
import time
from typing import Any, AsyncGenerator, Generator, Optional, Union

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import ToolMessage

from ols import config, constants
from ols.app.endpoints.ols import (
    calc_input_tokens,
    calc_output_tokens,
    consume_tokens,
    generate_response,
    get_available_quotas,
    log_processing_durations,
    process_request,
    store_conversation_history,
    store_transcript,
)
from ols.app.models.models import (
    Attachment,
    ErrorResponse,
    ForbiddenResponse,
    LLMRequest,
    RagChunk,
    ReferencedDocument,
    StreamedChunk,
    SummarizerResponse,
    TokenCounter,
    UnauthorizedResponse,
)
from ols.constants import MEDIA_TYPE_TEXT
from ols.customize import prompts
from ols.src.auth.auth import get_auth_dependency
from ols.utils import errors_parsing
from ols.utils.token_handler import PromptTooLongError

INVALID_QUERY_RESP = prompts.INVALID_QUERY_RESP

logger = logging.getLogger(__name__)

router = APIRouter(tags=["streaming_query"])
auth_dependency = get_auth_dependency(config.ols_config, virtual_path="/ols-access")


LLM_TOKEN_EVENT = "token"  # noqa: S105
LLM_TOOL_CALL_EVENT = "tool_call"
LLM_TOOL_RESULT_EVENT = "tool_result"


query_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Query is valid and stream/events from endpoint is returned",
        "model": str,
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
        "description": "Query can not be validated, LLM is not accessible or other internal error",
        "model": ErrorResponse,
    },
}


@router.post("/streaming_query", responses=query_responses)
def conversation_request(
    llm_request: LLMRequest,
    auth: Any = Depends(auth_dependency),
    user_id: Optional[str] = None,
) -> StreamingResponse:
    """Handle conversation requests for the OLS endpoint.

    Args:
        llm_request: The incoming request containing query details.
        auth: The authentication context, provided by dependency injection.
        user_id: Optional user ID used only when no-op auth is enabled.

    Returns:
        StreamingResponse: The streaming response generated for the query.
    """
    processed_request = process_request(auth, llm_request)

    summarizer_response: Union[
        AsyncGenerator[StreamedChunk, None], SummarizerResponse, Generator
    ]

    if not processed_request.valid:
        summarizer_response = invalid_response_generator()
    else:
        client_headers = llm_request.mcp_headers

        summarizer_response = generate_response(
            processed_request.conversation_id,
            llm_request,
            processed_request.previous_input,
            streaming=True,
            user_token=processed_request.user_token,
            client_headers=client_headers,
        )

    return StreamingResponse(
        response_processing_wrapper(
            summarizer_response,
            processed_request.user_id,
            processed_request.conversation_id,
            llm_request,
            processed_request.attachments,
            processed_request.valid,
            processed_request.query_without_attachments,
            llm_request.media_type,
            processed_request.timestamps,
            processed_request.skip_user_id_check,
        ),
        status_code=status.HTTP_200_OK,
        media_type=llm_request.media_type,
    )


async def invalid_response_generator() -> AsyncGenerator[StreamedChunk, None]:
    """Yield an invalid query response.

    Yields:
        str: The response indicating invalid query.
    """
    yield StreamedChunk(type="text", text=INVALID_QUERY_RESP)


def format_stream_data(d: dict) -> str:
    """Format outbound data in the Event Stream Format."""
    data = json.dumps(d)
    return f"data: {data}\n\n"


def stream_start_event(conversation_id: str) -> str:
    """Yield the start of the data stream.

    Args:
        conversation_id: The conversation ID (UUID).
    """
    return format_stream_data(
        {
            "event": "start",
            "data": {
                "conversation_id": conversation_id,
            },
        }
    )


def stream_event(data: dict, event_type: str, media_type: str) -> str:
    """Build an item to yield based on media type.

    Args:
        data: The data to yield.
        event_type: The type of event (e.g. token, tool request, tool execution).
        media_type: Media type of the response (e.g. text or JSON).

    Returns:
        str: The formatted string or JSON to yield.
    """
    if media_type == MEDIA_TYPE_TEXT:
        if event_type == LLM_TOKEN_EVENT:
            return data["token"]
        if event_type == LLM_TOOL_CALL_EVENT:
            return f"\nTool call: {json.dumps(data)}\n"
        if event_type == LLM_TOOL_RESULT_EVENT:
            return f"\nTool result: {json.dumps(data)}\n"
        logger.error("Unknown event type: %s", event_type)
        return ""
    return format_stream_data(
        {
            "event": event_type,
            "data": data,
        }
    )


def stream_end_event(
    ref_docs: list[dict],
    truncated: bool,
    media_type: str,
    token_counter: TokenCounter,
    available_quotas: dict[str, int],
) -> str:
    """Yield the end of the data stream.

    Args:
        ref_docs: Referenced documents.
        truncated: Indicates if the history was truncated.
        media_type: Media type of the response (e.g. text or JSON).
        token_counter: Token counter for the whole stream.
        available_quotas: Quotas available for configured quota limiters.
    """
    if media_type == constants.MEDIA_TYPE_JSON:
        return format_stream_data(
            {
                "event": "end",
                "data": {
                    "referenced_documents": ref_docs,
                    "truncated": truncated,
                    "input_tokens": calc_input_tokens(token_counter),
                    "output_tokens": calc_output_tokens(token_counter),
                },
                "available_quotas": available_quotas,
            }
        )
    ref_docs_string = "\n".join(
        f'{item["doc_title"]}: {item["doc_url"]}' for item in ref_docs
    )
    return f"\n\n---\n\n{ref_docs_string}" if ref_docs_string else ""


def build_referenced_docs(rag_chunks: list[RagChunk]) -> list[dict]:
    """Build a list of unique referenced documents."""
    referenced_documents = ReferencedDocument.from_rag_chunks(rag_chunks)
    return [
        {
            "doc_title": document.doc_title,
            "doc_url": document.doc_url,
        }
        for document in referenced_documents
    ]


def prompt_too_long_error(error: PromptTooLongError, media_type: str) -> str:
    """Return error representation for long prompts.

    Args:
        error: The exception raised for long prompts.
        media_type: Media type of the response (e.g. text or JSON).

    Returns:
        str: The error message formatted for the media type.
    """
    logger.error("Prompt is too long: %s", error)
    if media_type == MEDIA_TYPE_TEXT:
        return f"Prompt is too long: {error}"
    return format_stream_data(
        {
            "event": "error",
            "data": {
                "status_code": 413,
                "response": "Prompt is too long",
                "cause": str(error),
            },
        }
    )


def generic_llm_error(error: Exception, media_type: str) -> str:
    """Return error representation for generic LLM errors.

    Args:
        error: The exception raised during processing.
        media_type: Media type of the response (e.g. text or JSON).

    Returns:
        str: The error message formatted for the media type.
    """
    logger.error("Error while obtaining answer for user question")
    logger.exception(error)
    _, response, cause = errors_parsing.parse_generic_llm_error(error)

    response, cause = errors_parsing.handle_known_errors(response, cause)

    if media_type == MEDIA_TYPE_TEXT:
        return f"{response}: {cause}"
    return format_stream_data(
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
    tool_calls: list[dict],
    tool_results: list[ToolMessage],
    attachments: list[Attachment],
    valid: bool,
    query_without_attachments: str,
    rag_chunks: list[RagChunk],
    history_truncated: bool,
    timestamps: dict[str, float],
    skip_user_id_check: bool,
) -> None:
    """Store conversation history and transcript if enabled.

    Args:
        user_id: The user ID (UUID).
        conversation_id: The conversation ID (UUID).
        llm_request: The original request.
        response: The generated response.
        tool_calls: list of tool requests made in the query.
        tool_results: list of tool results from the query.
        attachments: list of attachments included in the query.
        valid: Indicates if the query was valid.
        query_without_attachments: Query content excluding attachments.
        rag_chunks: list of RAG (Retrieve-And-Generate) chunks used in the response.
        history_truncated: Indicates if the conversation history was truncated.
        timestamps: Dictionary tracking timestamps for various stages.
        skip_user_id_check: Skip user_id usid check.
    """
    store_conversation_history(
        user_id,
        conversation_id,
        llm_request,
        response,
        attachments,
        timestamps,
        skip_user_id_check,
        tool_calls=tool_calls,
        tool_results=tool_results,
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
            tool_calls,
            tool_results,
            attachments,
        )
    timestamps["store transcripts"] = time.time()


async def response_processing_wrapper(
    generator: AsyncGenerator[Any, None],
    user_id: str,
    conversation_id: str,
    llm_request: LLMRequest,
    attachments: list[Attachment],
    valid: bool,
    query_without_attachments: str,
    media_type: str,
    timestamps: dict[str, float],
    skip_user_id_check: bool,
) -> AsyncGenerator[str, None]:
    """Process the response from the generator and handle metadata and errors.

    Args:
        generator: The async generator providing summarizer responses.
        user_id: The user ID (UUID).
        conversation_id: The conversation ID (UUID).
        llm_request: The original request.
        attachments: list of attachments included in the query.
        valid: Indicates if the query was valid.
        query_without_attachments: Query content excluding attachments.
        media_type: Media type of the response (e.g. text or JSON).
        timestamps: Dictionary tracking timestamps for various stages.
        skip_user_id_check: Skip user_id usid check.

    Yields:
        str: The response items or error messages.
    """
    if media_type == constants.MEDIA_TYPE_JSON:
        yield stream_start_event(conversation_id)

    response: str = ""
    rag_chunks: list = []
    tool_calls: list = []
    tool_results: list = []
    history_truncated: bool = False
    idx: int = 0
    token_counter: Optional[TokenCounter] = None

    try:
        async for item in generator:
            if not isinstance(item, StreamedChunk):
                msg = f"Expecting StreamedChunk, but got {type(item)}: {item}"
                logger.error(msg)
                raise ValueError(msg)
            if item.type == "tool_call":
                tool_calls.append(item.data)
                yield stream_event(
                    data=item.data,
                    event_type=LLM_TOOL_CALL_EVENT,
                    media_type=media_type,
                )
            elif item.type == "tool_result":
                tool_results.append(item.data)
                yield stream_event(
                    data=item.data,
                    event_type=LLM_TOOL_RESULT_EVENT,
                    media_type=media_type,
                )
            elif item.type == "text":
                response += item.text
                yield stream_event(
                    data={"id": idx, "token": item.text},
                    event_type=LLM_TOKEN_EVENT,
                    media_type=media_type,
                )
                idx += 1
            elif item.type == "end":
                rag_chunks = item.data["rag_chunks"]
                history_truncated = item.data["truncated"]
                token_counter = item.data["token_counter"]
            else:
                msg = (
                    "Yielded unknown item type from streaming generator, "
                    f"item: {item}"
                )
                logger.error(msg)
                raise ValueError(msg)
    except PromptTooLongError as summarizer_error:
        yield prompt_too_long_error(summarizer_error, media_type)
        return  # stop execution after error

    except Exception as summarizer_error:
        yield generic_llm_error(summarizer_error, media_type)
        return  # stop execution after error

    timestamps["generate response"] = time.time()

    # Log assistant's answer in JSON format
    logger.info(
        json.dumps(
            {
                "event": "assistant_answer",
                "answer": response.strip(),
                "conversation_id": conversation_id,
                "user": user_id,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    store_data(
        user_id,
        conversation_id,
        llm_request,
        response,
        tool_calls,
        tool_results,
        attachments,
        valid,
        query_without_attachments,
        rag_chunks,
        history_truncated,
        timestamps,
        skip_user_id_check,
    )

    input_tokens = calc_input_tokens(token_counter)
    output_tokens = calc_output_tokens(token_counter)

    consume_tokens(
        config.quota_limiters,
        config.token_usage_history,
        user_id,
        input_tokens,
        output_tokens,
        llm_request.provider or config.ols_config.default_provider,
        llm_request.model or config.ols_config.default_model,
    )

    available_quotas = get_available_quotas(config.quota_limiters, user_id)

    yield stream_end_event(
        build_referenced_docs(rag_chunks),
        history_truncated,
        media_type,
        token_counter,
        available_quotas,
    )

    timestamps["add references"] = time.time()

    log_processing_durations(timestamps)
