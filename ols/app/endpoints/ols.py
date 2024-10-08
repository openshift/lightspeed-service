"""Handlers for all OLS-related REST API endpoints."""

import dataclasses
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pytz
from fastapi import APIRouter, Depends, HTTPException, status

from ols import config, constants
from ols.app import metrics
from ols.app.models.models import (
    Attachment,
    CacheEntry,
    ErrorResponse,
    ForbiddenResponse,
    LLMRequest,
    LLMResponse,
    PromptTooLongResponse,
    RagChunk,
    ReferencedDocument,
    SummarizerResponse,
    UnauthorizedResponse,
)
from ols.src.llms.llm_loader import LLMConfigurationError, resolve_provider_config
from ols.src.query_helpers.attachment_appender import append_attachments_to_query
from ols.src.query_helpers.docs_summarizer import DocsSummarizer
from ols.src.query_helpers.question_validator import QuestionValidator
from ols.utils import errors_parsing, suid
from ols.utils.auth_dependency import AuthDependency
from ols.utils.keywords import KEYWORDS
from ols.utils.token_handler import PromptTooLongError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])
auth_dependency = AuthDependency(virtual_path="/ols-access")


query_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Query is valid and correct response from LLM is returned",
        "model": LLMResponse,
    },
    401: {
        "description": "Missing or invalid credentials provided by client",
        "model": UnauthorizedResponse,
    },
    403: {
        "description": "Client does not have permission to access resource",
        "model": ForbiddenResponse,
    },
    413: {
        "description": "Prompt is too long",
        "model": PromptTooLongResponse,
    },
    500: {
        "description": "Query can not be validated, LLM is not accessible or other internal error",
        "model": ErrorResponse,
    },
}


@router.post("/query", responses=query_responses)
def conversation_request(
    llm_request: LLMRequest, auth: Any = Depends(auth_dependency)
) -> LLMResponse:
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
            conversation_id, llm_request, previous_input
        )

    timestamps["generate response"] = time.time()

    store_conversation_history(
        user_id, conversation_id, llm_request, summarizer_response.response, attachments
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
            summarizer_response.response,
            summarizer_response.rag_chunks,
            summarizer_response.history_truncated,
            attachments,
        )

    timestamps["store transcripts"] = time.time()

    # De-dup & retain order to create list of referenced documents
    referenced_documents = list(
        {
            rag_chunk.doc_url: ReferencedDocument(
                rag_chunk.doc_url,
                rag_chunk.doc_title,
            )
            for rag_chunk in summarizer_response.rag_chunks
        }.values()
    )

    timestamps["add references"] = time.time()
    log_processing_durations(timestamps)

    return LLMResponse(
        conversation_id=conversation_id,
        response=summarizer_response.response,
        referenced_documents=referenced_documents,
        truncated=summarizer_response.history_truncated,
    )


def log_processing_durations(timestamps: dict[str, float]) -> None:
    """Log processing durations."""

    def duration(key1: str, key2: str) -> float:
        """Calculate duration between two timestamps."""
        return timestamps[key2] - timestamps[key1]

    retrieve_user_duration = duration("start", "retrieve user")
    retrieve_conversation_duration = duration("retrieve user", "retrieve conversation")
    redact_query_duration = duration("retrieve conversation", "redact query")
    retrieve_previous_input_duration = duration(
        "redact query", "retrieve previous input"
    )
    append_attachmens_duration = duration(
        "retrieve previous input", "append attachments"
    )
    validate_question_duration = duration("append attachments", "validate question")
    generate_response_duration = duration("validate question", "generate response")
    store_transcripts_duration = duration("generate response", "store transcripts")
    add_references_duration = duration("store transcripts", "add references")
    total_duration = duration("start", "add references")

    # these messages can be grepped from logs and easily transformed into CSV file
    # for further processing and analysis
    msg = (
        f"Processing durations: {retrieve_user_duration},{retrieve_conversation_duration},"
        f"{redact_query_duration},{retrieve_previous_input_duration},{append_attachmens_duration},"
        f"{validate_question_duration},{generate_response_duration},{store_transcripts_duration},"
        f"{add_references_duration},{total_duration}"
    )

    logger.info(msg)


def retrieve_user_id(auth: Any) -> str:
    """Retrieve user ID from the token processed by auth. mechanism."""
    # auth contains tuple with user ID (in UUID format) and user name
    return auth[0]


def retrieve_conversation_id(llm_request: LLMRequest) -> str:
    """Retrieve conversation ID based on existing ID or on newly generated one."""
    conversation_id = llm_request.conversation_id

    # Generate a new conversation ID if not provided
    if not conversation_id:
        conversation_id = suid.get_suid()
        logger.info("%s New conversation", conversation_id)

    return conversation_id


def retrieve_previous_input(user_id: str, llm_request: LLMRequest) -> list[CacheEntry]:
    """Retrieve previous user input, if exists."""
    try:
        previous_input = []
        if llm_request.conversation_id:
            cache_content = config.conversation_cache.get(
                user_id, llm_request.conversation_id
            )
            if cache_content is not None:
                previous_input = cache_content
            logger.info(
                "%s Previous conversation input: %s",
                llm_request.conversation_id,
                previous_input,
            )
        return previous_input
    except Exception as e:
        logger.error("Error retrieving previous user input for user %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error retrieving conversation history",
                "cause": str(e),
            },
        )


def retrieve_attachments(llm_request: LLMRequest) -> list[Attachment]:
    """Retrieve attachments from the request."""
    attachments = llm_request.attachments

    # it is perfectly ok not to send any attachments
    if attachments is None:
        return []

    # some attachments were send to the service, time to check its metadata
    for attachment in attachments:
        if attachment.attachment_type not in constants.ATTACHMENT_TYPES:
            message = (
                f"Attachment with improper type {attachment.attachment_type} detected"
            )
            logger.error(message)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"response": "Unable to process this request", "cause": message},
            )
        if attachment.content_type not in constants.ATTACHMENT_CONTENT_TYPES:
            message = f"Attachment with improper content type {attachment.content_type} detected"
            logger.error(message)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"response": "Unable to process this request", "cause": message},
            )

    return attachments


def generate_response(
    conversation_id: str,
    llm_request: LLMRequest,
    previous_input: list[CacheEntry],
) -> SummarizerResponse:
    """Generate response based on validation result, previous input, and model output."""
    # Summarize documentation
    try:
        docs_summarizer = DocsSummarizer(
            provider=llm_request.provider, model=llm_request.model
        )
        history = CacheEntry.cache_entries_to_history(previous_input)
        return docs_summarizer.summarize(
            conversation_id, llm_request.query, config.rag_index, history
        )
    except PromptTooLongError as summarizer_error:
        logger.error("Prompt is too long: %s", summarizer_error)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "response": "Prompt is too long",
                "cause": str(summarizer_error),
            },
        )
    except Exception as summarizer_error:
        logger.error("Error while obtaining answer for user question")
        logger.exception(summarizer_error)
        status_code, response, cause = errors_parsing.parse_generic_llm_error(
            summarizer_error
        )
        raise HTTPException(
            status_code=status_code,
            detail={
                "response": response,
                "cause": cause,
            },
        )


def validate_requested_provider_model(llm_request: LLMRequest) -> None:
    """Validate provider/model; if provided in request payload."""
    provider = llm_request.provider
    model = llm_request.model
    if provider is None and model is None:
        # no validation, when provider & model are not sent with request
        return

    try:
        resolve_provider_config(provider, model, config.config.llm_providers)
    except LLMConfigurationError as e:
        metrics.llm_calls_validation_errors_total.inc()
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"response": "Unable to process this request", "cause": str(e)},
        )


def store_conversation_history(
    user_id: str,
    conversation_id: str,
    llm_request: LLMRequest,
    response: Optional[str],
    attachments: list[Attachment],
) -> None:
    """Store conversation history into selected cache.

    History is stored as simple dictionaries in the following format:
    ```python
    {"human_query": "texty", "ai_response": "text"},
    ```
    """
    try:
        if config.conversation_cache is not None:
            logger.info("%s Storing conversation history", conversation_id)
            cache_entry = CacheEntry(
                query=llm_request.query,
                response=response,
                attachments=attachments,
            )
            config.conversation_cache.insert_or_append(
                user_id,
                conversation_id,
                cache_entry,
            )
    except Exception as e:
        logger.error(
            "Error storing conversation history for user %s and conversation %s",
            user_id,
            conversation_id,
        )
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error storing conversation",
                "cause": str(e),
            },
        )


def redact_query(conversation_id: str, llm_request: LLMRequest) -> LLMRequest:
    """Redact query using query_redactor, raise HTTPException in case of any problem."""
    try:
        logger.debug("Redacting query for conversation %s", conversation_id)
        llm_request.query = config.query_redactor.redact(
            conversation_id, llm_request.query
        )
        return llm_request
    except Exception as redactor_error:
        logger.error(
            "Error while redacting query %s for conversation %s",
            redactor_error,
            conversation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error while redacting query",
                "cause": str(redactor_error),
            },
        )


def redact_attachments(
    conversation_id: str, attachments: list[Attachment]
) -> list[Attachment]:
    """Redact all attachments using query_redactor, raise HTTPException in case of any problem."""
    logger.debug("Redacting attachments for conversation %s", conversation_id)

    try:
        redacted_attachments = []
        for attachment in attachments:
            # might be possible to change attachments "in situ" but it might
            # confuse developers
            redacted_content = config.query_redactor.redact(
                conversation_id, attachment.content
            )
            redacted_attachment = Attachment(
                attachment_type=attachment.attachment_type,
                content_type=attachment.content_type,
                content=redacted_content,
            )
            redacted_attachments.append(redacted_attachment)
        return redacted_attachments

    except Exception as redactor_error:
        logger.error(
            "Error while redacting attachment %s for conversation %s",
            redactor_error,
            conversation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error while redacting attachment",
                "cause": str(redactor_error),
            },
        )


def _validate_question_llm(conversation_id: str, llm_request: LLMRequest) -> bool:
    """Validate user question using llm, raise HTTPException in case of any problem."""
    # Validate the query
    try:
        question_validator = QuestionValidator(
            provider=llm_request.provider, model=llm_request.model
        )
        return question_validator.validate_question(conversation_id, llm_request.query)
    except LLMConfigurationError as e:
        metrics.llm_calls_validation_errors_total.inc()
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"response": "Unable to process this request", "cause": str(e)},
        )
    except PromptTooLongError as e:
        logger.error("Prompt is too long: %s", e)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "response": "Prompt is too long",
                "cause": str(e),
            },
        )
    except Exception as validation_error:
        metrics.llm_calls_failures_total.inc()
        logger.error("Error while validating question")
        logger.exception(validation_error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error while validating question",
                "cause": str(validation_error),
            },
        )


def _validate_question_keyword(query: str) -> bool:
    """Validate user question using keyword."""
    # Current implementation is without any tokenizer method, lemmatization/n-grams.
    # Add valid keywords to keywords.py file.
    query_temp = query.lower()
    for kw in KEYWORDS:
        if kw in query_temp:
            return True
    # query_temp = {q_word.lower().strip(".?,") for q_word in query.split()}
    # common_words = config.keywords.intersection(query_temp)
    # if len(common_words) > 0:
    #     return constants.SUBJECT_ALLOWED

    logger.debug("No matching keyword found for query: %s", query)
    return False


def validate_question(conversation_id: str, llm_request: LLMRequest) -> bool:
    """Validate user question."""
    match config.ols_config.query_validation_method:

        case constants.QueryValidationMethod.LLM:
            logger.debug("LLM based query validation.")
            return _validate_question_llm(conversation_id, llm_request)

        case constants.QueryValidationMethod.KEYWORD:
            logger.debug("Keyword based query validation.")
            return _validate_question_keyword(llm_request.query)

        case _:
            # Query validation disabled by default
            logger.debug(
                "%s Question validation is disabled. Treating question as valid.",
                conversation_id,
            )
            return True


def construct_transcripts_path(user_id: str, conversation_id: str) -> Path:
    """Construct path to transcripts."""
    # these two normalizations are required by Snyk as it detects
    # this Path sanitization pattern
    uid = os.path.normpath("/" + user_id).lstrip("/")
    cid = os.path.normpath("/" + conversation_id).lstrip("/")
    return Path(
        config.ols_config.user_data_collection.transcripts_storage,
        uid,
        cid,
    )


def store_transcript(
    user_id: str,
    conversation_id: str,
    query_is_valid: bool,
    redacted_query: str,
    llm_request: LLMRequest,
    response: str,
    rag_chunks: list[RagChunk],
    truncated: bool,
    attachments: list[Attachment],
) -> None:
    """Store transcript in the local filesystem.

    Args:
        user_id: The user ID (UUID).
        conversation_id: The conversation ID (UUID).
        query_is_valid: The result of the query validation.
        redacted_query: The redacted query (without attachments).
        llm_request: The request containing a query.
        response: The response to store.
        rag_chunks: The list of `RagChunk` objects.
        truncated: The flag indicating if the history was truncated.
        attachments: The list of `Attachment` objects.
    """
    # Creates transcripts path only if it doesn't exist. The `exist_ok=True` prevents
    # race conditions in case of multiple server instances trying to set up transcripts
    # at the same location.
    transcripts_path = construct_transcripts_path(user_id, conversation_id)
    transcripts_path.mkdir(parents=True, exist_ok=True)

    data_to_store = {
        "metadata": {
            "provider": llm_request.provider or config.ols_config.default_provider,
            "model": llm_request.model or config.ols_config.default_model,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "timestamp": datetime.now(pytz.UTC).isoformat(),
        },
        "redacted_query": redacted_query,
        "query_is_valid": query_is_valid,
        "llm_response": response,
        "rag_chunks": [dataclasses.asdict(rag_chunk) for rag_chunk in rag_chunks],
        "truncated": truncated,
        "attachments": [attachment.model_dump() for attachment in attachments],
    }

    # stores feedback in a file under unique uuid
    transcript_file_path = transcripts_path / f"{suid.get_suid()}.json"
    with open(transcript_file_path, "w", encoding="utf-8") as transcript_file:
        json.dump(data_to_store, transcript_file)

    logger.debug("transcript stored in '%s'", transcript_file_path)
