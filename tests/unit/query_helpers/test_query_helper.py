"""Unit tests for query helper class."""

from ols.src.llms.llm_loader import load_llm
from ols.src.query_helpers.query_helper import QueryHelper
from ols.utils.config import ConfigManager


def test_defaults_used():
    """Test that the defaults are used when no inputs are provided."""
    ConfigManager._instance = None
    config_manager = ConfigManager()
    config_manager.init_config("tests/config/valid_config.yaml")

    qh = QueryHelper()

    assert qh.provider == config_manager.get_ols_config().default_provider
    assert qh.model == config_manager.get_ols_config().default_model
    assert qh.llm_loader is load_llm
    assert qh.llm_params == {}


def test_inputs_are_used():
    """Test that the inputs are used when provided."""
    test_provider = "test_provider"
    test_model = "test_model"
    qh = QueryHelper(provider=test_provider, model=test_model)

    assert qh.provider == test_provider
    assert qh.model == test_model
