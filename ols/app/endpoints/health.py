"""Handlers for OLS health REST API endpoints.

These endpoints are used to check if service is live and prepared to accept
requests. Note that these endpoints can be accessed using GET or HEAD HTTP
methods. For HEAD HTTP method, just the HTTP response code is used.
"""

import logging

from fastapi import APIRouter
from langchain_core.messages.ai import AIMessage

from ols import config
from ols.app.models.models import LivenessResponse, ReadinessResponse
from ols.src.llms.llm_loader import load_llm

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)
llm_is_ready_persistent_state = False


def llm_is_ready() -> bool:
    """Check if the LLM can be loaded and is ready to serve.

    If so, store the success to `llm_is_ready_persistent_state` to cache
    the result for future calls.
    """
    global llm_is_ready_persistent_state
    if llm_is_ready_persistent_state is True:
        return True
    try:
        bare_llm = load_llm(
            config.ols_config.default_provider, config.ols_config.default_model
        )
        response = bare_llm.invoke(input="Hello there!")
        # BAM and Watsonx replies as str and not as `AIMessage`
        if isinstance(response, (str, AIMessage)):
            logger.info("LLM connection checked - LLM is ready")
            llm_is_ready_persistent_state = True
            return True
        raise ValueError(f"Unexpected response from LLM: {response}")
    except Exception as e:
        logger.error(f"LLM connection check failed with - {e}")
        return False


def index_is_ready() -> bool:
    """Check if the index is loaded."""
    if config._rag_index is None and config.ols_config.reference_content is not None:
        return False
    return True


@router.get("/readiness")
def readiness_probe_get_method() -> ReadinessResponse:
    """Ready status of service."""
    if not index_is_ready():
        return ReadinessResponse(ready=False, reason="index is not ready")
    if not llm_is_ready():
        return ReadinessResponse(ready=False, reason="LLM is not ready")
    return ReadinessResponse(ready=True, reason="service is ready")


@router.get("/liveness")
def liveness_probe_get_method() -> LivenessResponse:
    """Live status of service."""
    return LivenessResponse(alive=True)
