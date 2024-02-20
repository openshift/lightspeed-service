"""Unit tests for query helper class."""

from ols.src.llms.llm_loader import LLMLoader
from ols.src.query_helpers.query_helper import QueryHelper
from ols.utils import config


def test_defaults_used():
    """Test that the defaults are used when no inputs are provided."""
    config.init_config("tests/config/valid_config.yaml")

    qh = QueryHelper()

    assert qh.provider == config.ols_config.default_provider
    assert qh.model == config.ols_config.default_model
    assert qh.llm_loader == LLMLoader
    assert qh.llm_params == {}


def test_inputs_are_used():
    """Test that the inputs are used when provided."""
    test_provider = "test_provider"
    test_model = "test_model"
    qh = QueryHelper(provider=test_provider, model=test_model)

    assert qh.provider == test_provider
    assert qh.model == test_model
