"""Unit tests for query helper class."""

from ols import config
from ols.src.llms.llm_loader import load_llm
from ols.src.query_helpers.query_helper import QueryHelper


def test_defaults_used():
    """Test that the defaults are used when no inputs are provided."""
    config.reload_from_yaml_file("tests/config/valid_config.yaml")

    qh = QueryHelper()

    assert qh.provider == config.ols_config.default_provider
    assert qh.model == config.ols_config.default_model
    assert qh.llm_loader is load_llm
    assert qh.generic_llm_params == {}
    assert qh.streaming is False


def test_inputs_are_used():
    """Test that the inputs are used when provided."""
    test_provider = "test_provider"
    test_model = "test_model"
    qh = QueryHelper(provider=test_provider, model=test_model)

    assert qh.provider == test_provider
    assert qh.model == test_model


def test_streaming_parameter():
    """Test that the optional streaming parameter is stored."""
    qh = QueryHelper(streaming=False)
    assert qh.streaming is False

    qh = QueryHelper(streaming=True)
    assert qh.streaming is True
