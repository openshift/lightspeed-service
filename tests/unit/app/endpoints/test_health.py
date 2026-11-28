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
from tests.unit.utils.test_audit_support import audit_records as audit_records
from tests.unit.utils.test_audit_support import records


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


def test_readiness_probe_llm_check__str_msg():
    """Test succesfull scenario - LLM responds as str."""
    with (
        patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False),
        patch("ols.app.endpoints.health.load_llm") as mocked_load_llm,
    ):
        mocked_load_llm.return_value = MockedLLM(invoke_return="message")
        assert llm_is_ready()


def test_readiness_probe_llm_check__ai_msg():
    """Test succesfull scenario - LLM responds as AIMessage."""
    with (
        patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False),
        patch("ols.app.endpoints.health.load_llm") as mocked_load_llm,
    ):
        mocked_load_llm.return_value = MockedLLM(invoke_return=AIMessage("message"))
        assert llm_is_ready()


def test_readiness_probe_llm_check__llm_unexpected_response():
    """Test fail scenario - llm responds in unexpected format."""
    with (
        patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False),
        patch("ols.app.endpoints.health.load_llm") as mocked_load_llm,
    ):
        mocked_load_llm.return_value = MockedLLM(invoke_return={"nobody": "knows"})
        assert not llm_is_ready()


def test_readiness_probe_llm_check__state_cache():
    """Test succesfull scenario - LLM check is done only once."""
    with (
        patch("ols.config._conversation_cache", create=True, new=mock_cache()),
        patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False),
        patch("ols.app.endpoints.health.load_llm") as mocked_load_llm,
    ):
        mocked_load_llm.return_value = MockedLLM(invoke_return="message")
        assert llm_is_ready()
        assert mocked_load_llm.call_count == 1

        response = readiness_probe_get_method()
        assert response == ReadinessResponse(ready=True, reason="service is ready")

        # try again and check if the llm function was invoked again - it shoudn't
        llm_is_ready()
        assert mocked_load_llm.call_count == 1


def test_readiness_probe_llm_check__state_cache_not_expired():
    """Test the scenario with cache not expired - LLM check is done only once."""
    with (
        patch("ols.config._conversation_cache", create=True, new=mock_cache()),
        patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False),
        patch("ols.app.endpoints.health.load_llm") as mocked_load_llm,
    ):
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


def test_readiness_probe_llm_check__state_cache_expired():
    """Test the scenario with cache expired - LLM check is done twice."""
    with (
        patch("ols.config._conversation_cache", create=True, new=mock_cache()),
        patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False),
        patch("ols.app.endpoints.health.load_llm") as mocked_load_llm,
    ):
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


def test_readiness_probe_llm_check__llm_raise():
    """Test fail scenario - llm raises an exception (eg. invalid credentials)."""
    with (
        patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False),
        patch("ols.app.endpoints.health.load_llm") as mocked_load_llm,
    ):
        mocked_load_llm.side_effect = Exception
        assert not llm_is_ready()


def test_readiness_probe_get_method_service_is_ready():
    """Test the readiness_probe function when the service is ready."""
    with (
        patch("ols.config._conversation_cache", create=True, new=mock_cache()),
        patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=False),
        patch("ols.app.endpoints.health.load_llm") as mocked_load_llm,
    ):
        mocked_load_llm.return_value = MockedLLM(invoke_return="message")

        response = readiness_probe_get_method()
        assert response == ReadinessResponse(ready=True, reason="service is ready")


def test_readiness_probe_get_method_index_is_ready():
    """Test the readiness_probe function when index is loaded."""
    with (
        patch("ols.config._conversation_cache", create=True, new=mock_cache()),
        patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=True),
    ):
        # simulate that the index is loaded
        config._rag_index_loader = Mock(vector_indexes=["idx"])
        config.ols_config.reference_content = Mock(indexes=["idx_cfg"])
        assert index_is_ready()
        response = readiness_probe_get_method()
        assert response == ReadinessResponse(ready=True, reason="service is ready")

        # simulate that the index is not loaded, but it shouldn't as there
        # is no reference content in config
        config._rag_index_loader = None
        config.ols_config.reference_content = None
        assert index_is_ready()
        response = readiness_probe_get_method()
        assert response == ReadinessResponse(ready=True, reason="service is ready")

        # reference_content present but no indexes — should still be ready
        config._rag_index_loader = None
        config.ols_config.reference_content = Mock(indexes=None)
        assert index_is_ready()

        config.ols_config.reference_content = Mock(indexes=[])
        assert index_is_ready()


def test_readiness_probe_get_method_index_not_ready():
    """Test the readiness_probe function when index is not loaded."""
    with patch("ols.app.endpoints.health.llm_is_ready_persistent_state", new=True):
        config._rag_index_loader = Mock(vector_indexes=None)
        config.ols_config.reference_content = Mock(indexes=["idx_cfg"])

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


def test_readiness_audits_real_worker_attempt_not_future_timeout(
    audit_records, monkeypatch
):
    """Keep health purpose and native attempt lifetime after the caller times out."""
    import threading

    import httpx
    from langchain_openai import ChatOpenAI

    from ols.app.endpoints import health
    from ols.utils.audit_support import audit_model

    entered, release, done = threading.Event(), threading.Event(), threading.Event()

    def handler(_request):
        entered.set()
        assert release.wait(5), "readiness test did not release controlled I/O"
        return httpx.Response(
            200,
            json={
                "id": "chat-test",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = audit_model(
            ChatOpenAI(
                model="test", api_key="unused", max_retries=0, http_client=client
            )
        )

        def invoke(prompt):
            try:
                return model.invoke(prompt)
            finally:
                done.set()

        bare = Mock()
        bare.invoke.side_effect = invoke
        monkeypatch.setattr(health, "load_llm", Mock(return_value=bare))
        monkeypatch.setattr(health, "llm_is_ready_persistent_state", False)
        monkeypatch.setattr(health, "LLM_READINESS_TIMEOUT", 0.1)
        try:
            assert not health.llm_is_ready()
            assert entered.is_set()
            assert len(records(audit_records)) == 1
            assert "trace_id" not in records(audit_records)[0]
        finally:
            release.set()
            assert done.wait(5)
        started, completed = records(audit_records)
        assert started["attempt_id"] == completed["attempt_id"]
        assert completed["outcome"] == "success"
        assert health.llm_is_ready_persistent_state is False
        monkeypatch.setattr(health, "llm_is_ready_persistent_state", True)
        config.ols_config.expire_llm_is_ready_persistent_state = -1
        assert health.llm_is_ready()
        assert len(records(audit_records)) == 2
        bare.invoke.assert_called_once_with("Hello there!")
