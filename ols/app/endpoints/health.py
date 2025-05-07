"""Handlers for OLS health REST API endpoints.

These endpoints are used to check if service is live and prepared to accept
requests. Note that these endpoints can be accessed using GET or HEAD HTTP
methods. For HEAD HTTP method, just the HTTP response code is used.
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, status
from langchain_core.messages.ai import AIMessage

from ols import config
from ols.app.models.models import (
    LivenessResponse,
    NotAvailableResponse,
    ReadinessResponse,
)
from ols.src.llms.llm_loader import load_llm

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)
llm_is_ready_persistent_state: bool = False
llm_is_ready_timestamp = 0  # pylint: disable=C0103


def llm_is_ready() -> bool:
    """Check if the LLM can be loaded and is ready to serve.

    If so, store the success to `llm_is_ready_persistent_state` to cache
    the result for future calls.
    """
    global llm_is_ready_persistent_state, llm_is_ready_timestamp  # pylint: disable=global-statement
    last_called, llm_is_ready_timestamp = llm_is_ready_timestamp, int(time.time())
    if llm_is_ready_persistent_state is True and (
        not config.ols_config.expire_llm_is_ready_persistent_state
        or config.ols_config.expire_llm_is_ready_persistent_state < 0
        or (llm_is_ready_timestamp - last_called)
        < config.ols_config.expire_llm_is_ready_persistent_state
    ):
        return True
    # Reset `llm_is_ready_persistent_state`
    llm_is_ready_persistent_state = False
    try:
        bare_llm = load_llm(
            config.ols_config.default_provider,
            config.ols_config.default_model,
        )
        response = bare_llm.invoke(input="Hello there!")
        # BAM and Watsonx replies as str and not as `AIMessage`
        if isinstance(response, (str, AIMessage)):
            logger.info("LLM connection checked - LLM is ready")
            llm_is_ready_persistent_state = True
            return True
        raise ValueError(f"Unexpected response from LLM: {response}")
    except Exception as e:
        logger.error("LLM connection check failed with - %s", e)
        return False


def index_is_ready() -> bool:
    """Check if the index is loaded."""
    if config.rag_index is None and config.ols_config.reference_content is not None:
        return False
    return True


def cache_is_ready() -> bool:
    """Check if the cache is ready."""
    if config.conversation_cache is None:
        return False
    return config.conversation_cache.ready()


get_readiness_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Service is ready",
        "model": ReadinessResponse,
    },
    503: {
        "description": "Service is not ready",
        "model": NotAvailableResponse,
    },
}


@router.get("/readiness", responses=get_readiness_responses)
def readiness_probe_get_method() -> ReadinessResponse:
    """Ready status of service."""
    if not index_is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "response": "Service is not ready",
                "cause": "Index is not ready",
            },
        )
    if not llm_is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "response": "Service is not ready",
                "cause": "LLM is not ready",
            },
        )
    if not cache_is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "response": "Service is not ready",
                "cause": "Cache is not ready",
            },
        )

    return ReadinessResponse(ready=True, reason="service is ready")


get_liveness_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Service is alive",
        "model": LivenessResponse,
    },
}


@router.get("/liveness", responses=get_liveness_responses)
def liveness_probe_get_method() -> LivenessResponse:
    """Live status of service."""
    return LivenessResponse(alive=True)
