"""Unit tests for health endpoints handlers."""

import time
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException
from langchain_core.messages.ai import AIMessage

from ols import config
from ols.app.endpoints.health import (
    index_is_ready,
    liveness_probe_get_method,
    llm_is_ready,
    readiness_probe_get_method,
)
from ols.app.models.config import InMemoryCacheConfig
from ols.app.models.models import LivenessResponse, ReadinessResponse
from ols.src.cache.in_memory_cache import InMemoryCache


def mock_cache():
    """Fixture with constructed and initialized in memory cache object."""
    mc = InMemoryCacheConfig({"max_entries": "10"})
    mc_kls = Mock(InMemoryCache)
    c = mc_kls(mc)
    c.initialize_cache(mc)
    return c


class MockedLLM:
    """Mocked LLM object that returns the same value on every invoke call."""

    def __init__(self, invoke_return):
        """Initialize the object with the return value for `invoke` method."""
        self.invoke_return = invoke_return

    def invoke(self, *args, **kwargs):
        """Return the value set in the constructor."""
        return self.invoke_return


@patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False)
@patch("ols.app.endpoints.health.load_llm")
def test_readiness_probe_llm_check__str_msg(mocked_load_llm):
    """Test succesfull scenario - LLM responds as str."""
    mocked_load_llm.return_value = MockedLLM(invoke_return="message")
    assert llm_is_ready()


@patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False)
@patch("ols.app.endpoints.health.load_llm")
def test_readiness_probe_llm_check__ai_msg(mocked_load_llm):
    """Test succesfull scenario - LLM responds as AIMessage."""
    mocked_load_llm.return_value = MockedLLM(invoke_return=AIMessage("message"))
    assert llm_is_ready()


@patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False)
@patch("ols.app.endpoints.health.load_llm")
def test_readiness_probe_llm_check__llm_unexpected_response(mocked_load_llm):
    """Test fail scenario - llm responds in unexpected format."""
    mocked_load_llm.return_value = MockedLLM(invoke_return={"nobody": "knows"})
    assert not llm_is_ready()


@patch("ols.config._conversation_cache", create=True, new=mock_cache())
@patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False)
@patch("ols.app.endpoints.health.load_llm")
def test_readiness_probe_llm_check__state_cache(mocked_load_llm):
    """Test succesfull scenario - LLM check is done only once."""
    mocked_load_llm.return_value = MockedLLM(invoke_return="message")
    assert llm_is_ready()
    assert mocked_load_llm.call_count == 1

    response = readiness_probe_get_method()
    assert response == ReadinessResponse(ready=True, reason="service is ready")

    # try again and check if the llm function was invoked again - it shoudn't
    llm_is_ready()
    assert mocked_load_llm.call_count == 1


@patch("ols.config._conversation_cache", create=True, new=mock_cache())
@patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False)
@patch("ols.app.endpoints.health.load_llm")
def test_readiness_probe_llm_check__state_cache_not_expired(mocked_load_llm):
    """Test the scenario with cache not expired - LLM check is done only once."""
    try:
        # Set cache expiration time to 1 sec.
        config.ols_config.expire_llm_is_ready_persistent_state = 1
        mocked_load_llm.return_value = MockedLLM(invoke_return="message")
        assert llm_is_ready()
        assert mocked_load_llm.call_count == 1

        response = readiness_probe_get_method()
        assert response == ReadinessResponse(ready=True, reason="service is ready")

        # try again and check if the llm function was invoked again - it shouldn't
        llm_is_ready()
        assert mocked_load_llm.call_count == 1
    finally:
        # Reset the expire_llm_is_ready_persistent_state option.
        config.ols_config.expire_llm_is_ready_persistent_state = -1


@patch("ols.config._conversation_cache", create=True, new=mock_cache())
@patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False)
@patch("ols.app.endpoints.health.load_llm")
def test_readiness_probe_llm_check__state_cache_expired(mocked_load_llm):
    """Test the scenario with cache expired - LLM check is done twice."""
    try:
        # Set cache expiration time to 1 sec.
        config.ols_config.expire_llm_is_ready_persistent_state = 1
        mocked_load_llm.return_value = MockedLLM(invoke_return="message")
        assert llm_is_ready()
        assert mocked_load_llm.call_count == 1

        response = readiness_probe_get_method()
        assert response == ReadinessResponse(ready=True, reason="service is ready")
        # Wait for 1.5 secs and let the cache get expired.
        time.sleep(1.5)

        # try again and check if the llm function was invoked again - it should.
        llm_is_ready()
        assert mocked_load_llm.call_count == 2
    finally:
        # Reset the expire_llm_is_ready_persistent_state option.
        config.ols_config.expire_llm_is_ready_persistent_state = -1


@patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False)
@patch("ols.app.endpoints.health.load_llm")
def test_readiness_probe_llm_check__llm_raise(mocked_load_llm):
    """Test fail scenario - llm raises an exception (eg. invalid credentials)."""
    mocked_load_llm.side_effect = Exception
    assert not llm_is_ready()


@patch("ols.config._conversation_cache", create=True, new=mock_cache())
@patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False)
@patch("ols.app.endpoints.health.load_llm")
def test_readiness_probe_get_method_service_is_ready(mocked_load_llm):
    """Test the readiness_probe function when the service is ready."""
    mocked_load_llm.return_value = MockedLLM(invoke_return="message")

    response = readiness_probe_get_method()
    assert response == ReadinessResponse(ready=True, reason="service is ready")


@patch("ols.config._conversation_cache", create=True, new=mock_cache())
@patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=True)
def test_readiness_probe_get_method_index_is_ready():
    """Test the readiness_probe function when index is loaded."""
    # simulate that the index is loaded
    config._rag_index = True
    assert index_is_ready()
    response = readiness_probe_get_method()
    assert response == ReadinessResponse(ready=True, reason="service is ready")

    # simulate that the index is not loaded, but it shouldn't as there
    # is no reference content in config
    config._rag_index = None
    config.ols_config.reference_content = None
    assert index_is_ready()
    response = readiness_probe_get_method()
    assert response == ReadinessResponse(ready=True, reason="service is ready")


@patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=True)
def test_readiness_probe_get_method_index_not_ready():
    """Test the readiness_probe function when index is not loaded."""
    # simulate that the index is not loaded
    config._rag_index = None
    config.ols_config.reference_content = "something else than None"

    assert not index_is_ready()
    with pytest.raises(HTTPException, match="Service is not ready"):
        readiness_probe_get_method()


def test_readiness_probe_get_method_cache_not_ready():
    """Test the readiness_probe function when cache is not ready."""
    # simulate that the cache is not ready
    with patch("ols.config._conversation_cache", new=None) as mocked_cache:
        with pytest.raises(HTTPException, match="Service is not ready"):
            readiness_probe_get_method()

    with patch(
        "ols.config._conversation_cache", create=True, new=mock_cache()
    ) as mocked_cache:
        mocked_cache.ready.return_value = False
        with pytest.raises(HTTPException, match="Service is not ready"):
            readiness_probe_get_method()


def test_liveness_probe_get_method():
    """Test the liveness_probe function."""
    # the tested function returns constant right now
    # i.e. it does not depend on application state
    response = liveness_probe_get_method()
    assert response == LivenessResponse(alive=True)
