"""Handlers for all OLS-related REST API endpoints."""

import dataclasses
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional, Union

import psycopg2
import pytz
from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import AIMessage, HumanMessage

from ols import config, constants
from ols.app import metrics
from ols.app.models.models import (
    Attachment,
    CacheEntry,
    ErrorResponse,
    ForbiddenResponse,
    LLMRequest,
    LLMResponse,
    ProcessedRequest,
    PromptTooLongResponse,
    RagChunk,
    ReferencedDocument,
    SummarizerResponse,
    TokenCounter,
    UnauthorizedResponse,
)
from ols.customize import keywords, prompts
from ols.src.auth.auth import get_auth_dependency
from ols.src.llms.llm_loader import LLMConfigurationError, resolve_provider_config
from ols.src.query_helpers.attachment_appender import append_attachments_to_query
from ols.src.query_helpers.docs_summarizer import DocsSummarizer
from ols.src.query_helpers.question_validator import QuestionValidator
from ols.src.quota.quota_limiter import QuotaLimiter
from ols.src.quota.token_usage_history import TokenUsageHistory
from ols.utils import errors_parsing, suid
from ols.utils.token_handler import PromptTooLongError

KEYWORDS = keywords.KEYWORDS
INVALID_QUERY_RESP = prompts.INVALID_QUERY_RESP

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])
auth_dependency = get_auth_dependency(config.ols_config, virtual_path="/ols-access")

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


def merge_tools_info(tool_calls: list[dict], tool_results: list[dict]) -> list[dict]:
    """Merge tools calls and results into a single list of dictionaries."""
    # check if tool_calls and tool_results contains information we
    # expect - these are highly unlikely scenarios, but if it happens
    # we want to log it instead of failing with status code to client
    if len(tool_calls) != len(tool_results):
        logger.error("tool_calls and tool_results must have the same number of items")
        return []
    if len(tool_calls) != len({i["id"] for i in tool_calls}):
        logger.error("tool_calls must have unique ids")
        return []
    if len(tool_results) != len({i["id"] for i in tool_results}):
        logger.error("tool_results must have unique ids")
        return []
    if {i["id"] for i in tool_calls} != {i["id"] for i in tool_results}:
        logger.error("tool_calls and tool_results must have the same number of ids")
        return []

    # convert tool_results to a dictionary for quick lookup
    tools_results_dict = {item["id"]: item for item in tool_results}

    # merge lists based on 'id'
    merged_tool_info = [
        {**item, **tools_results_dict[item["id"]]}
        for item in tool_calls
        if item["id"] in tools_results_dict
    ]

    return merged_tool_info


@router.post("/query", responses=query_responses)
def conversation_request(
    llm_request: LLMRequest,
    auth: Any = Depends(auth_dependency),
    user_id: Optional[str] = None,
) -> LLMResponse:
    """Handle conversation requests for the OLS endpoint.

    Args:
        llm_request: The request containing a query, conversation ID, and optional attachments.
        auth: The Authentication handler (FastAPI Depends) that will handle authentication Logic.
        user_id: Optional user ID used only when no-op auth is enabled.

    Returns:
        Response containing the processed information.
    """
    processed_request = process_request(auth, llm_request)

    summarizer_response: SummarizerResponse | Generator

    if not processed_request.valid:
        # response containing info about query that can not be validated
        summarizer_response = SummarizerResponse(
            INVALID_QUERY_RESP,
            [],
            False,
            None,
        )
    else:
        summarizer_response = generate_response(
            processed_request.conversation_id,
            llm_request,
            processed_request.previous_input,
            streaming=False,
            user_token=processed_request.user_token,
        )

    processed_request.timestamps["generate response"] = time.time()

    # Log assistant's answer in JSON format
    logger.info(
        json.dumps(
            {
                "event": "assistant_answer",
                "answer": summarizer_response.response.strip(),
                "conversation_id": processed_request.conversation_id,
                "user": processed_request.user_id,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # Log tool calls for non-streaming endpoint
    for tool_call in summarizer_response.tool_calls:
        logger.info(
            json.dumps(
                {
                    "event": "tool_call",
                    "tool_name": tool_call.get("name", "unknown"),
                    "arguments": tool_call.get("args", {}),
                    "tool_id": tool_call.get("id", "unknown"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    # Log tool results for non-streaming endpoint
    for tool_result in summarizer_response.tool_results:
        logger.info(
            json.dumps(
                {
                    "event": "tool_result",
                    "tool_id": tool_result.get("id", "unknown"),
                    "status": tool_result.get("status", "unknown"),
                    "output_snippet": str(tool_result.get("content", ""))[
                        :1000
                    ],  # Truncate to first 1000 chars
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    store_conversation_history(
        processed_request.user_id,
        processed_request.conversation_id,
        llm_request,
        summarizer_response.response,
        processed_request.attachments,
        processed_request.timestamps,
        processed_request.skip_user_id_check,
    )

    if config.ols_config.user_data_collection.transcripts_disabled:
        logger.debug("transcripts collections is disabled in configuration")
    else:
        store_transcript(
            processed_request.user_id,
            processed_request.conversation_id,
            processed_request.valid,
            processed_request.query_without_attachments,
            llm_request,
            summarizer_response.response,
            summarizer_response.rag_chunks,
            summarizer_response.history_truncated,
            summarizer_response.tool_calls,
            summarizer_response.tool_results,
            processed_request.attachments,
        )

    processed_request.timestamps["store transcripts"] = time.time()

    referenced_documents = ReferencedDocument.from_rag_chunks(
        summarizer_response.rag_chunks
    )

    processed_request.timestamps["add references"] = time.time()
    log_processing_durations(processed_request.timestamps)

    input_tokens = calc_input_tokens(summarizer_response.token_counter)
    output_tokens = calc_output_tokens(summarizer_response.token_counter)

    consume_tokens(
        config.quota_limiters,
        config.token_usage_history,
        processed_request.user_id,
        input_tokens,
        output_tokens,
        llm_request.provider or config.ols_config.default_provider,
        llm_request.model or config.ols_config.default_model,
    )

    available_quotas = get_available_quotas(
        config.quota_limiters, processed_request.user_id
    )

    return LLMResponse(
        conversation_id=processed_request.conversation_id,
        response=summarizer_response.response,
        referenced_documents=referenced_documents,
        truncated=summarizer_response.history_truncated,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        available_quotas=available_quotas,
        tool_calls=summarizer_response.tool_calls,
        tool_results=summarizer_response.tool_results,
    )


def calc_input_tokens(token_counter: Optional[TokenCounter]) -> int:
    """Calculate input tokens."""
    if token_counter is None:
        return 0
    return token_counter.input_tokens


def calc_output_tokens(token_counter: Optional[TokenCounter]) -> int:
    """Calculate output tokens."""
    if token_counter is None:
        return 0
    return token_counter.output_tokens


def get_available_quotas(
    quota_limiters: Optional[list[QuotaLimiter]],
    user_id: str,
) -> dict[str, int]:
    """Get quota available from all quota limiters."""
    available_quotas: dict[str, int] = {}
    # check if any quota limiter is configured
    if quota_limiters is not None:
        for quota_limiter in quota_limiters:
            name = quota_limiter.__class__.__name__
            available_quota = quota_limiter.available_quota(user_id)
            available_quotas[name] = available_quota
    return available_quotas


def consume_tokens(
    quota_limiters: Optional[list[QuotaLimiter]],
    token_usage_history: Optional[TokenUsageHistory],
    user_id: str,
    input_tokens: int,
    output_tokens: int,
    provider: str,
    model: str,
) -> None:
    """Consume tokens from cluster and/or user quotas."""
    # check if token usage history is configured
    if token_usage_history is not None:
        token_usage_history.consume_tokens(
            user_id, provider, model, input_tokens, output_tokens
        )

    # check if any quota limiter is configured
    if quota_limiters is not None:
        for quota_limiter in quota_limiters:
            quota_limiter.consume_tokens(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                subject_id=user_id,
            )


def process_request(auth: Any, llm_request: LLMRequest) -> ProcessedRequest:
    """Process incoming request.

    Args:
        auth: The Authentication handler (FastAPI Depends) that will handle authentication Logic.
        llm_request: The request containing a query, conversation ID, and optional attachments.

    Returns:
        Tuple containing the processed information.
        User ID, conversation ID, query without attachments, previous input,
        attachments, validation result, timestamps, skip_user_id_check and user token
    """
    timestamps = {"start": time.time()}

    user_id = retrieve_user_id(auth)
    logger.info("Auth module: %s", config.ols_config.authentication_config.module)
    logger.info("User ID: %s", user_id)
    timestamps["retrieve user"] = time.time()

    conversation_id = retrieve_conversation_id(llm_request)
    timestamps["retrieve conversation"] = time.time()

    skip_user_id_check = retrieve_skip_user_id_check(auth)

    user_token = retrieve_user_token(auth)

    # Important note: Redact the query before attempting to do any
    # logging of the query to avoid leaking PII into logs.

    # Redact the query
    llm_request = redact_query(conversation_id, llm_request)
    timestamps["redact query"] = time.time()

    # Log incoming request (after redaction) in JSON format
    logger.info(
        json.dumps(
            {
                "event": "user_question",
                "question": llm_request.query,
                "user": user_id,
                "conversation_id": conversation_id,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    previous_input = retrieve_previous_input(
        user_id, llm_request.conversation_id, skip_user_id_check
    )
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

    check_tokens_available(config.quota_limiters, user_id)

    # Validate the query
    if not previous_input:
        valid = validate_question(conversation_id, llm_request)
    else:
        logger.debug("follow-up conversation - skipping question validation")
        valid = True

    timestamps["validate question"] = time.time()

    return ProcessedRequest(
        user_id=user_id,
        conversation_id=conversation_id,
        query_without_attachments=query_without_attachments,
        previous_input=previous_input,
        attachments=attachments,
        valid=valid,
        timestamps=timestamps,
        skip_user_id_check=skip_user_id_check,
        user_token=user_token,
    )


def check_tokens_available(
    quota_limiters: Optional[list[QuotaLimiter]], user_id: str
) -> None:
    """Check if tokens are available for user."""
    # no quota limiters specified
    if quota_limiters is None:
        return

    try:
        for quota_limiter in quota_limiters:
            quota_limiter.ensure_available_quota(subject_id=user_id)
    except psycopg2.Error as pg_error:
        message = "Error communicating with quota database backend"
        logger.error(message)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": message,
                "cause": str(pg_error),
            },
        )
    except Exception as quota_exceed_error:
        message = "The quota has been exceeded"
        logger.error(message)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": message,
                "cause": str(quota_exceed_error),
            },
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


def retrieve_skip_user_id_check(auth: Any) -> bool:
    """Retrieve skip user_id check from the token processed by auth. mechanism."""
    return auth[2]


def retrieve_user_token(auth: Any) -> str:
    """Retrieve user token from the token processed by auth. mechanism."""
    return auth[3]


def retrieve_conversation_id(llm_request: LLMRequest) -> str:
    """Retrieve conversation ID based on existing ID or on newly generated one."""
    conversation_id = llm_request.conversation_id

    # Generate a new conversation ID if not provided
    if not conversation_id:
        conversation_id = suid.get_suid()
        logger.info("%s New conversation", conversation_id)

    return conversation_id


def retrieve_previous_input(
    user_id: str, conversation_id: str, skip_user_id_check: bool = False
) -> list[CacheEntry]:
    """Retrieve previous user input, if exists."""
    try:
        previous_input = []
        if conversation_id:
            cache_content = config.conversation_cache.get(
                user_id, conversation_id, skip_user_id_check
            )
            if cache_content is not None:
                previous_input = cache_content
            logger.info(
                "Conversation ID: %s Previous conversation input: %s",
                conversation_id,
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
    streaming: bool = False,
    user_token: Optional[str] = None,
) -> Union[SummarizerResponse, Generator]:
    """Generate response based on validation result, previous input, and model output.

    Args:
        conversation_id: The unique identifier for the conversation.
        llm_request: The request containing a query.
        previous_input: The history of the conversation (if available).
        streaming: The flag indicating if the response should be streamed.
        user_token: The user token used for authorization.

    Returns:
        SummarizerResponse or Generator, depending on the streaming flag.
    """
    try:
        docs_summarizer = DocsSummarizer(
            provider=llm_request.provider,
            model=llm_request.model,
            system_prompt=llm_request.system_prompt,
            user_token=user_token,
        )
        history = CacheEntry.cache_entries_to_history(previous_input)
        if streaming:
            return docs_summarizer.generate_response(
                llm_request.query, config.rag_index_loader.get_retriever(), history
            )
        response = docs_summarizer.create_response(
            llm_request.query,
            config.rag_index_loader.get_retriever(),
            history,
        )
        logger.debug("%s Generated response: %s", conversation_id, response)
        return response
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
        status_code, response_text, cause = errors_parsing.parse_generic_llm_error(
            summarizer_error
        )
        response_text, cause = errors_parsing.handle_known_errors(response_text, cause)
        raise HTTPException(
            status_code=status_code,
            detail={
                "response": response_text,
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
    timestamps: dict[str, float],
    skip_user_id_check: bool = False,
) -> None:
    """Store conversation history into selected cache.

    History is stored as simple dictionaries in the following format:
    ```python
    {"human_query": "texty", "ai_response": "text"},
    ```
    """
    try:
        if response is None:
            response = ""
        if config.conversation_cache is not None:
            logger.info("%s Storing conversation history", conversation_id)
            query_message = HumanMessage(content=llm_request.query)
            response_message = AIMessage(content=response)
            if timestamps:
                query_message.response_metadata = {"created_at": timestamps["start"]}
                response_message.response_metadata["created_at"] = timestamps[
                    "generate response"
                ]
            if llm_request.provider:
                response_message.response_metadata["provider"] = llm_request.provider
            if llm_request.model:
                response_message.response_metadata["model"] = llm_request.model

            cache_entry = CacheEntry(
                query=query_message,
                response=response_message,
                attachments=attachments,
            )
            config.conversation_cache.insert_or_append(
                user_id,
                conversation_id,
                cache_entry,
                skip_user_id_check,
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
            provider=llm_request.provider,
            model=llm_request.model,
            system_prompt=llm_request.system_prompt,
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
    tool_calls: list[dict],
    tool_results: list[dict],
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
        tool_calls: The list of tool requests.
        tool_results: The list of tool results.
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
        "tool_calls": merge_tools_info(tool_calls, tool_results),
        "attachments": [attachment.model_dump() for attachment in attachments],
    }

    # stores feedback in a file under unique uuid
    transcript_file_path = transcripts_path / f"{suid.get_suid()}.json"
    with open(transcript_file_path, "w", encoding="utf-8") as transcript_file:
        json.dump(data_to_store, transcript_file)

    logger.debug("transcript stored in '%s'", transcript_file_path)
