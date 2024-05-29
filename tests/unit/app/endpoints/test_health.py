"""Unit tests for health endpoints handlers."""

from unittest.mock import patch

from langchain_core.messages.ai import AIMessage

from ols import config
from ols.app.endpoints.health import (
    index_is_ready,
    liveness_probe_get_method,
    llm_is_ready,
    readiness_probe_get_method,
)
from ols.app.models.models import LivenessResponse, ReadinessResponse


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


@patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False)
@patch("ols.app.endpoints.health.load_llm")
def test_readiness_probe_llm_check__llm_raise(mocked_load_llm):
    """Test fail scenario - llm raises an exception (eg. invalid credentials)."""
    mocked_load_llm.side_effect = Exception
    assert not llm_is_ready()


@patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False)
@patch("ols.app.endpoints.health.load_llm")
def test_readiness_probe_get_method_service_is_ready(mocked_load_llm):
    """Test the readiness_probe function when the service is ready."""
    mocked_load_llm.return_value = MockedLLM(invoke_return="message")

    response = readiness_probe_get_method()
    assert response == ReadinessResponse(ready=True, reason="service is ready")


def test_readiness_probe_get_method_index_is_ready():
    """Test the readiness_probe function when index is loaded."""
    # simulate that the index is loaded
    config._rag_index = True
    assert index_is_ready()
    response = readiness_probe_get_method()
    assert response == ReadinessResponse(ready=False, reason="LLM is not ready")

    # simulate that the index is not loaded, but it shouldn't as there
    # is no reference content in config
    config._rag_index = None
    config.ols_config.reference_content = None
    assert index_is_ready()
    response = readiness_probe_get_method()
    assert response == ReadinessResponse(ready=False, reason="LLM is not ready")


def test_readiness_probe_get_method_index_not_ready():
    """Test the readiness_probe function when index is not loaded."""
    # simulate that the index is not loaded
    config._rag_index = None
    config.ols_config.reference_content = "something else than None"

    assert not index_is_ready()
    response = readiness_probe_get_method()
    assert response == ReadinessResponse(ready=False, reason="index is not ready")


def test_liveness_probe_get_method():
    """Test the liveness_probe function."""
    # the tested function returns constant right now
    # i.e. it does not depend on application state
    response = liveness_probe_get_method()
    assert response == LivenessResponse(alive=True)
