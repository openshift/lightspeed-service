"""Handlers for all OLS-related REST API endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from ols import constants
from ols.app import metrics
from ols.app.models.models import LLMRequest, LLMResponse
from ols.src.llms.llm_loader import LLMConfigurationError, LLMLoader
from ols.src.query_helpers.docs_summarizer import DocsSummarizer
from ols.src.query_helpers.question_validator import QuestionValidator
from ols.utils import config, suid

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


@router.post("/query")
def conversation_request(llm_request: LLMRequest) -> LLMResponse:
    """Handle conversation requests for the OLS endpoint.

    Args:
        llm_request: The request containing a query and conversation ID.

    Returns:
        Response containing the processed information.
    """
    # Initialize variables
    previous_input = None

    # TODO: retrieve proper user ID from request
    user_id = "user1"

    conversation_id = retrieve_conversation_id(llm_request)

    # TODO: will be refactored in following PR
    if llm_request.conversation_id:
        previous_input = config.conversation_cache.get(user_id, conversation_id)
        logger.info(f"{conversation_id} Previous conversation input: {previous_input}")

    # Log incoming request
    logger.info(f"{conversation_id} Incoming request: {llm_request.query}")

    # Validate the query
    try:
        question_validator = QuestionValidator(
            provider=llm_request.provider, model=llm_request.model
        )
        metrics.llm_calls_total.inc()
        validation_result = question_validator.validate_question(
            conversation_id, llm_request.query
        )
    except LLMConfigurationError as e:
        metrics.llm_calls_validation_errors_total.inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"response": f"Unable to process this request because '{e}'"},
        )
    except Exception as validation_error:
        metrics.llm_calls_failures_total.inc()
        logger.error("Error while validating question")
        logger.error(validation_error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error while validating question",
        )

    response: Optional[str] = None

    match (validation_result):
        case constants.SUBJECT_INVALID:
            logger.info(
                f"{conversation_id} - Query is not relevant to kubernetes or ocp, returning"
            )
            response = (
                "I can only answer questions about OpenShift and Kubernetes. "
                "Please rephrase your question"
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
                llm_response, _ = docs_summarizer.summarize(
                    conversation_id, llm_request.query, previous_input
                )
                # TODO: There are some inconsistencies in the types to be solved.
                # See the comment in `summarize` method in `docs_summarizer.py`
                # Because of that, we are ignoring some type checks when we
                # are creating response model.
                response = llm_response.response
            except Exception as summarizer_error:
                logger.error("Error while obtaining answer for user question")
                logger.error(summarizer_error)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Error while obtaining answer for user question",
                )

    if config.conversation_cache is not None:
        config.conversation_cache.insert_or_append(
            user_id,
            conversation_id,
            llm_request.query + "\n\n" + str(response or ""),
        )
    return LLMResponse(conversation_id=conversation_id, response=response)


def retrieve_conversation_id(llm_request: LLMRequest) -> str:
    """Retrieve conversation ID based on existing ID or on newly generated one."""
    conversation_id = llm_request.conversation_id

    # Generate a new conversation ID if not provided
    if not conversation_id:
        conversation_id = suid.get_suid()
        logger.info(f"{conversation_id} New conversation")

    return conversation_id


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

    bare_llm = LLMLoader(
        config.ols_config.default_provider,
        config.ols_config.default_model,
    ).llm

    prompt = PromptTemplate.from_template("{query}")
    llm_chain = LLMChain(llm=bare_llm, prompt=prompt, verbose=True)
    response = llm_chain(inputs={"query": llm_request.query})

    logger.info(f"{conversation_id} Model returned: {response}")

    llm_response = LLMResponse(
        conversation_id=conversation_id, response=response["text"]
    )

    return llm_response
