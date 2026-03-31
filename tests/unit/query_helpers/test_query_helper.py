"""Unit tests for query helper class."""

from ols import config
from ols.constants import QueryMode
from ols.src.llms.llm_loader import load_llm
from ols.src.prompts.prompts import (
    QUERY_SYSTEM_INSTRUCTION,
    TROUBLESHOOTING_SYSTEM_INSTRUCTION,
)
from ols.src.query_helpers.query_helper import QueryHelper


def test_defaults_used():
    """Test that the defaults are used when no inputs are provided."""
    config.reload_from_yaml_file("tests/config/valid_config.yaml")

    qh = QueryHelper()

    assert qh.provider == config.ols_config.default_provider
    assert qh.model == config.ols_config.default_model
    assert qh.llm_loader is load_llm
    assert qh.generic_llm_params == {}


def test_inputs_are_used():
    """Test that the inputs are used when provided."""
    test_provider = "test_provider"
    test_model = "test_model"
    qh = QueryHelper(provider=test_provider, model=test_model)

    assert qh.provider == test_provider
    assert qh.model == test_model


def test_ask_mode_uses_query_system_instruction():
    """Test that ASK mode selects QUERY_SYSTEM_INSTRUCTION as the system prompt."""
    config.reload_from_yaml_file("tests/config/valid_config.yaml")
    config.ols_config.system_prompt = None

    qh = QueryHelper(mode=QueryMode.ASK)

    assert qh._system_prompt == QUERY_SYSTEM_INSTRUCTION


def test_troubleshooting_mode_uses_troubleshooting_instruction():
    """Test that TROUBLESHOOTING mode selects TROUBLESHOOTING_SYSTEM_INSTRUCTION."""
    config.reload_from_yaml_file("tests/config/valid_config.yaml")
    config.ols_config.system_prompt = None

    qh = QueryHelper(mode=QueryMode.TROUBLESHOOTING)

    assert qh._system_prompt == TROUBLESHOOTING_SYSTEM_INSTRUCTION


def test_config_system_prompt_takes_precedence_over_mode():
    """Test that a config-level system prompt overrides the mode default."""
    config.reload_from_yaml_file("tests/config/valid_config.yaml")
    config.ols_config.system_prompt = "custom prompt from config"

    qh = QueryHelper(mode=QueryMode.TROUBLESHOOTING)

    assert qh._system_prompt == "custom prompt from config"

    config.ols_config.system_prompt = None


def test_default_mode_is_ask():
    """Test that omitting mode defaults to ASK behavior."""
    config.reload_from_yaml_file("tests/config/valid_config.yaml")
    config.ols_config.system_prompt = None

    qh_default = QueryHelper()
    qh_ask = QueryHelper(mode=QueryMode.ASK)

    assert qh_default._system_prompt == qh_ask._system_prompt


def test_mode_stored_as_instance_attribute():
    """Test that mode is stored as self._mode on QueryHelper."""
    config.reload_from_yaml_file("tests/config/valid_config.yaml")

    qh_ask = QueryHelper(mode=QueryMode.ASK)
    assert qh_ask._mode == QueryMode.ASK

    qh_ts = QueryHelper(mode=QueryMode.TROUBLESHOOTING)
    assert qh_ts._mode == QueryMode.TROUBLESHOOTING

    qh_default = QueryHelper()
    assert qh_default._mode == QueryMode.ASK
