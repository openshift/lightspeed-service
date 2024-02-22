"""Handlers for all OLS-related REST API endpoints."""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from ols import constants
from ols.app import metrics
from ols.app.models.models import LLMRequest, LLMResponse
from ols.src.llms.llm_loader import LLMConfigurationError, load_llm
from ols.src.query_helpers.docs_summarizer import DocsSummarizer
from ols.src.query_helpers.question_validator import QuestionValidator
from ols.utils import config, suid
from ols.utils.auth_dependency import auth_dependency

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


@router.post("/query")
def conversation_request(
    llm_request: LLMRequest, auth: Any = Depends(auth_dependency)
) -> LLMResponse:
    """Handle conversation requests for the OLS endpoint.

    Args:
        llm_request: The request containing a query and conversation ID.
        auth: The Authentication handler (FastAPI Depends) that will handle authentication Logic.

    Returns:
        Response containing the processed information.
    """
    # Initialize variables
    previous_input = None
    referenced_documents: list[str] = []

    # TODO: retrieve proper user ID from request
    user_id = "user1"

    conversation_id = retrieve_conversation_id(llm_request)
    previous_input = retrieve_previous_input(user_id, llm_request)

    # Log incoming request
    logger.info(f"{conversation_id} Incoming request: {llm_request.query}")

    metrics.llm_calls_total.inc()

    # Redact the query
    llm_request = redact_query(conversation_id, llm_request)

    # Validate the query
    if not previous_input:
        validation_result = validate_question(conversation_id, llm_request)
    else:
        logger.debug("follow-up conversation - skipping question validation")
        validation_result = constants.SUBJECT_VALID

    response, referenced_documents, truncated = generate_response(
        conversation_id, llm_request, validation_result, previous_input
    )

    store_conversation_history(user_id, conversation_id, llm_request, response)
    return LLMResponse(
        conversation_id=conversation_id,
        response=response,
        referenced_documents=referenced_documents,
        truncated=truncated,
    )


@router.post("/debug/query")
def conversation_request_debug_api(llm_request: LLMRequest) -> LLMResponse:
    """Handle requests for the base LLM completion endpoint.

    Args:
        llm_request: The request containing a query.

    Returns:
        Response containing the processed information.
    """
    conversation_id = retrieve_conversation_id(llm_request)
    logger.info(f"{conversation_id} Incoming request: {llm_request.query}")

    response = generate_bare_response(conversation_id, llm_request)

    return LLMResponse(
        conversation_id=conversation_id,
        response=response,
        referenced_documents=[],
        truncated=False,
    )


def retrieve_conversation_id(llm_request: LLMRequest) -> str:
    """Retrieve conversation ID based on existing ID or on newly generated one."""
    conversation_id = llm_request.conversation_id

    # Generate a new conversation ID if not provided
    if not conversation_id:
        conversation_id = suid.get_suid()
        logger.info(f"{conversation_id} New conversation")

    return conversation_id


def retrieve_previous_input(user_id: str, llm_request: LLMRequest) -> Optional[str]:
    """Retrieve previous user input, if exists."""
    previous_input = None
    if llm_request.conversation_id:
        previous_input = config.conversation_cache.get(
            user_id, llm_request.conversation_id
        )
        logger.info(
            f"{llm_request.conversation_id} Previous conversation input: {previous_input}"
        )
    return previous_input


def generate_response(
    conversation_id: str,
    llm_request: LLMRequest,
    validation_result: str,
    previous_input: Optional[str],
) -> tuple[Optional[str], list[str], bool]:
    """Generate response based on validation result, previous input, and model output."""
    match (validation_result):
        case constants.SUBJECT_INVALID:
            logger.info(
                f"{conversation_id} - Query is not relevant to kubernetes or ocp, returning"
            )
            return (
                (
                    "I can only answer questions about OpenShift and Kubernetes. "
                    "Please rephrase your question"
                ),
                [],
                False,
            )
        case constants.SUBJECT_VALID:
            logger.info(
                f"{conversation_id} - Question is relevant to kubernetes or ocp"
            )
            # Summarize documentation
            try:
                docs_summarizer = DocsSummarizer(
                    provider=llm_request.provider, model=llm_request.model
                )
                llm_response = docs_summarizer.summarize(
                    conversation_id, llm_request.query, previous_input
                )
                return (
                    llm_response["response"],
                    llm_response["referenced_documents"],
                    llm_response["history_truncated"],
                )
            except Exception as summarizer_error:
                logger.error("Error while obtaining answer for user question")
                logger.exception(summarizer_error)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Error while obtaining answer for user question",
                )
        case _:
            logger.error("Invalid validation result (internal error)")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error while obtaining answer for user question",
            )


def store_conversation_history(
    user_id: str, conversation_id: str, llm_request: LLMRequest, response: Optional[str]
) -> None:
    """Store conversation history into selected cache."""
    if config.conversation_cache is not None:
        logger.info(f"{conversation_id} Storing conversation history.")
        config.conversation_cache.insert_or_append(
            user_id,
            conversation_id,
            llm_request.query + "\n\n" + str(response or ""),
        )


def redact_query(conversation_id: str, llm_request: LLMRequest) -> LLMRequest:
    """Redact query using query_redactor, raise HTTPException in case of any problem."""
    try:
        logger.debug(f"Redacting query for conversation {conversation_id}")
        if not config.query_redactor:
            logger.debug("query_redactor not found")
            return llm_request
        llm_request.query = config.query_redactor.redact_query(
            conversation_id, llm_request.query
        )
        return llm_request
    except Exception as redactor_error:
        logger.error(
            f"Error while redacting query {redactor_error} for conversation {conversation_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"response": f"Error while redacting query '{redactor_error}'"},
        )


def validate_question(conversation_id: str, llm_request: LLMRequest) -> str:
    """Validate user question, raise HTTPException in case of any problem."""
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
            detail={"response": f"Unable to process this request because '{e}'"},
        )
    except Exception as validation_error:
        metrics.llm_calls_failures_total.inc()
        logger.error("Error while validating question")
        logger.exception(validation_error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error while validating question",
        )


def generate_bare_response(conversation_id: str, llm_request: LLMRequest) -> str:
    """Generate bare response without validation not using conversation history."""
    bare_llm = load_llm(
        config.ols_config.default_provider,
        config.ols_config.default_model,
    )

    prompt = PromptTemplate.from_template("{query}")
    llm_chain = LLMChain(llm=bare_llm, prompt=prompt, verbose=True)
    response = llm_chain(inputs={"query": llm_request.query})

    logger.info(f"{conversation_id} Model returned: {response}")
    return response["text"]
