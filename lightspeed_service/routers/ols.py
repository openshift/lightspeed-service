import os

from fastapi import APIRouter
from fastapi import HTTPException

import lightspeed_service.constants as constants
from lightspeed_service.cache.cache_factory import CacheFactory
from lightspeed_service.docs.docs_summarizer import DocsSummarizer
from lightspeed_service.models import LLMRequest
from lightspeed_service.query_helpers.happy_response_generator import (
    HappyResponseGenerator,
)
from lightspeed_service.query_helpers.question_validator import (
    QuestionValidator,
)
from lightspeed_service.query_helpers.yaml_generator import YamlGenerator
from lightspeed_service.utils.logger import Logger
from lightspeed_service.utils.model_context import get_watsonx_predictor
from lightspeed_service.utils.suid import get_suid

router = APIRouter(prefix="/ols", tags=["ols"])


@router.post("")
def ols_request(llm_request: LLMRequest):
    """
    Handle requests for the OLS endpoint.

    Args:
        llm_request (LLMRequest): The request containing a query and
            conversation ID.

    Returns:
        dict: Response containing the processed information.
    """
    conversation_cache = CacheFactory.conversation_cache()
    logger = Logger("ols_endpoint").logger

    # Initialize variables
    previous_input = None
    conversation = llm_request.conversation_id

    # Generate a new conversation ID if not provided
    if conversation is None:
        conversation = get_suid()
        logger.info(f"{conversation} New conversation")
    else:
        previous_input = conversation_cache.get(conversation)
        logger.info(
            f"{conversation} Previous conversation input: {previous_input}"
        )

    llm_response = LLMRequest(
        query=llm_request.query, conversation_id=conversation
    )

    # Log incoming request
    logger.info(f"{conversation} Incoming request: {llm_request.query}")

    # Validate the query
    question_validator = QuestionValidator()
    validation_result = question_validator.validate_question(
        conversation, llm_request.query
    )

    if validation_result[0] == constants.INVALID:
        logger.info(f"{conversation} Question is not about k8s/ocp, rejecting")
        raise HTTPException(
            status_code=422,
            detail={
                "response": "Sorry, I can only answer questions about "
                "OpenShift and Kubernetes. This does not look "
                "like something I know how to handle."
            },
        )

    if validation_result[0] == constants.VALID:
        logger.info(f"{conversation} Question is about k8s/ocp")

        # Generate a user-friendly response wrapper
        response_wrapper = HappyResponseGenerator()
        wrapper = response_wrapper.generate(conversation, llm_request.query)

        if validation_result[1] == constants.NOYAML:
            logger.info(
                f"{conversation} Question is not about yaml, sending for "
                "generic info"
            )

            # Summarize documentation
            docs_summarizer = DocsSummarizer()
            summary, _ = docs_summarizer.summarize(
                conversation, llm_request.query
            )

            llm_response.response = wrapper + "\n" + summary
            return llm_response

        elif validation_result[1] == constants.YAML:
            logger.info(
                f"{conversation} Question is about yaml, sending to the YAML "
                "generator"
            )
            yaml_generator = YamlGenerator()
            generated_yaml = yaml_generator.generate_yaml(
                conversation, llm_request.query, previous_input
            )

            if generated_yaml == constants.SOME_FAILURE:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "response": "Internal server error. Please try again."
                    },
                )

            # Further processing of YAML response (filtering, cleaning,
            # linting, RAG, etc.)

            llm_response.response = wrapper + "\n" + generated_yaml
            conversation_cache.insert_or_append(
                conversation,
                llm_request.query + "\n\n" + llm_response.response,
            )
            return llm_response

        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "response": "Internal server error. Please try again."
                },
            )
    else:
        raise HTTPException(
            status_code=500,
            detail={"response": "Internal server error. Please try again."},
        )


@router.post("/raw_prompt")
@router.post("/base_llm_completion")
def base_llm_completion(llm_request: LLMRequest):
    """
    Handle requests for the base LLM completion endpoint.

    Args:
        llm_request (LLMRequest): The request containing a query.

    Returns:
        dict: Response containing the processed information.
    """
    base_completion_model = os.getenv(
        "BASE_COMPLETION_MODEL", "ibm/granite-20b-instruct-v1"
    )
    logger = Logger("base_llm_completion_endpoint").logger
    if llm_request.conversation_id is None:
        conversation = get_suid()
    else:
        conversation = llm_request.conversation_id

    llm_response = LLMRequest(query=llm_request.query)
    llm_response.conversation_id = conversation

    logger.info(f"{conversation} New conversation")
    logger.info(f"{conversation} Incoming request: {llm_request.query}")

    bare_llm = get_watsonx_predictor(model=base_completion_model)
    response = bare_llm(llm_request.query)

    # TODO: Make the removal of endoftext some kind of function
    clean_response = response.split("<|endoftext|>")[0]
    llm_response.response = clean_response

    logger.info(f"{conversation} Model returned: {llm_response.response}")

    return llm_response
