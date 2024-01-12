"""Unit tests for YamlGenerator class."""

import pytest

import ols.src.query_helpers.yaml_generator
from ols.src.query_helpers.yaml_generator import YamlGenerator
from ols.utils import config
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader


@pytest.fixture
def yaml_generator():
    """Fixture containing constructed and initialized YamlGenerator."""
    config.load_empty_config()
    return YamlGenerator()


def test_yaml_generator(yaml_generator, monkeypatch):
    """Test the basic functionality of YAML generator."""
    ml = mock_llm_chain({"text": "default"})
    monkeypatch.setattr(ols.src.query_helpers.yaml_generator, "LLMChain", ml)
    monkeypatch.setattr(
        ols.src.query_helpers.yaml_generator, "LLMLoader", mock_llm_loader()
    )

    # response will be constant because we mocked LLMChain and LLMLoader
    response = yaml_generator.generate_yaml("1234", "the query")
    assert response == "default"


def test_yaml_generator_history_enabled(yaml_generator, monkeypatch):
    """Test the basic functionality of YAML generator when history is enabled."""
    history_value = "conversation history"

    def llmchain_param_check(self, *args, **kwargs):
        """Modified version of __call__ special method for LLMChain mock."""
        assert "inputs" in kwargs, "Missing keyword argument 'input'"
        inputs = kwargs["inputs"]
        assert "history" in inputs, f"Missing attribute 'history' in {inputs}"
        assert inputs["history"] == history_value
        return {"text": "default"}

    # use mocked LLMChain
    ml = mock_llm_chain(None)
    monkeypatch.setattr(ols.src.query_helpers.yaml_generator, "LLMChain", ml)
    # modify __call__ special method with more checks
    ml.__call__ = llmchain_param_check

    monkeypatch.setattr(
        ols.src.query_helpers.yaml_generator, "LLMLoader", mock_llm_loader()
    )

    # response will be constant because we mocked LLMChain and LLMLoader
    response = yaml_generator.generate_yaml(
        "1234", "the query", history="conversation history"
    )
    assert response == "default"


def test_yaml_generator_verbose_enabled(yaml_generator, monkeypatch):
    """Test the basic functionality of YAML generator when verbosity is enabled."""

    def llmchain_param_check(self, *args, **kwargs):
        """Modified version of __init__ special method for LLMChain mock."""
        assert "verbose" in kwargs, f"Missing attribute 'verbose' in {kwargs}"

    # use mocked LLMChain
    ml = mock_llm_chain({"text": "default"})
    monkeypatch.setattr(ols.src.query_helpers.yaml_generator, "LLMChain", ml)
    # modify __init__ special method with more checks
    ml.__init__ = llmchain_param_check

    monkeypatch.setattr(
        ols.src.query_helpers.yaml_generator, "LLMLoader", mock_llm_loader()
    )

    # response will be constant because we mocked LLMChain and LLMLoader
    response = yaml_generator.generate_yaml("1234", "the query", verbose="true")
    assert response == "default"
