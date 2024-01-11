"""Unit tests for class HappyResponseGenerator."""

import pytest

import ols.src.query_helpers.happy_response_generator
from ols.src.query_helpers.happy_response_generator import HappyResponseGenerator
from ols.utils import config
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader


@pytest.fixture
def happy_response_generator():
    """Fixture containing constructed and initialized HappyResponseGenerator."""
    config.load_empty_config()
    return HappyResponseGenerator()


def test_generate(happy_response_generator, monkeypatch):
    """Basic test for happy response generator."""
    ml = mock_llm_chain({"text": "default"})
    monkeypatch.setattr(ols.src.query_helpers.happy_response_generator, "LLMChain", ml)
    monkeypatch.setattr(
        ols.src.query_helpers.happy_response_generator, "LLMLoader", mock_llm_loader()
    )

    # everything is mocked so response will remain the same
    response = happy_response_generator.generate("1234", "question")
    assert response == "default"


def test_generate_empty_question(happy_response_generator, monkeypatch):
    """Basic test for happy response generator."""
    ml = mock_llm_chain({"text": "default"})
    monkeypatch.setattr(ols.src.query_helpers.happy_response_generator, "LLMChain", ml)
    monkeypatch.setattr(
        ols.src.query_helpers.happy_response_generator, "LLMLoader", mock_llm_loader()
    )

    # everything is mocked so response will remain the same
    response = happy_response_generator.generate("1234", "")
    assert response == "default"
