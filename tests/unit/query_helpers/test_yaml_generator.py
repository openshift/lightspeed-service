"""Unit tests for YamlGenerator class."""

from unittest.mock import patch

import pytest

from ols.src.query_helpers.yaml_generator import QueryHelper, YamlGenerator
from ols.utils import config
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader


@pytest.fixture
def yaml_generator():
    """Fixture containing constructed and initialized YamlGenerator."""
    config.init_empty_config()
    return YamlGenerator()


def test_is_query_helper_subclass():
    """Test that YamlGenerator is a subclass of QueryHelper."""
    assert issubclass(YamlGenerator, QueryHelper)


@patch("ols.src.query_helpers.yaml_generator.LLMLoader", new=mock_llm_loader(None))
def test_yaml_generator(yaml_generator):
    """Test the basic functionality of YAML generator."""
    ml = mock_llm_chain({"text": "default"})
    with patch("ols.src.query_helpers.yaml_generator.LLMChain", new=ml):
        # response will be constant because we mocked LLMChain and LLMLoader
        response = yaml_generator.generate_yaml("1234", "the query")
        assert response == "default"


@patch("ols.src.query_helpers.yaml_generator.LLMLoader", new=mock_llm_loader(None))
def test_yaml_generator_history_enabled(yaml_generator):
    """Test the basic functionality of YAML generator when history is enabled."""
    history_value = "conversation history"

    def llmchain_param_check(self, *args, **kwargs):
        """Modify __call__ special method for LLMChain mock."""
        assert "inputs" in kwargs, "Missing keyword argument 'input'"
        inputs = kwargs["inputs"]
        assert "history" in inputs, f"Missing attribute 'history' in {inputs}"
        assert inputs["history"] == history_value
        return {"text": "default"}

    # use mocked LLMChain
    ml = mock_llm_chain(None)
    with patch("ols.src.query_helpers.yaml_generator.LLMChain", new=ml):
        # modify __call__ special method with more checks
        ml.__call__ = llmchain_param_check

        # response will be constant because we mocked LLMChain and LLMLoader
        response = yaml_generator.generate_yaml(
            "1234", "the query", history="conversation history"
        )
        assert response == "default"


@patch("ols.src.query_helpers.yaml_generator.LLMLoader", new=mock_llm_loader(None))
def test_yaml_generator_verbose_enabled(yaml_generator):
    """Test the basic functionality of YAML generator when verbosity is enabled."""

    def llmchain_param_check(self, *args, **kwargs):
        """Modify __init__ special method for LLMChain mock."""
        assert "verbose" in kwargs, f"Missing attribute 'verbose' in {kwargs}"

    # use mocked LLMChain
    ml = mock_llm_chain({"text": "default"})
    with patch("ols.src.query_helpers.yaml_generator.LLMChain", new=ml):
        # modify __init__ special method with more checks
        ml.__init__ = llmchain_param_check

        # response will be constant because we mocked LLMChain and LLMLoader
        response = yaml_generator.generate_yaml("1234", "the query", verbose="true")
        assert response == "default"
