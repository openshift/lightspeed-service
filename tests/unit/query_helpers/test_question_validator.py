"""Unit tests for QuestionValidator class."""

import pytest

import ols.src.query_helpers.question_validator
from ols.src.query_helpers.question_validator import QuestionValidator
from ols.utils import config
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader


@pytest.fixture
def question_validator():
    """Fixture containing constructed and initialized QuestionValidator."""
    config.load_empty_config()
    return QuestionValidator()


def test_invalid_response(question_validator, monkeypatch):
    """Test how invalid responses are handled by QuestionValidator."""
    # response not in the following set should generate a ValueError
    # [INVALID,NOYAML]
    # [VALID,NOYAML]
    # [VALID,YAML]

    ml = mock_llm_chain({"text": "default"})
    monkeypatch.setattr(ols.src.query_helpers.question_validator, "LLMChain", ml)
    monkeypatch.setattr(
        ols.src.query_helpers.question_validator, "LLMLoader", mock_llm_loader()
    )

    with pytest.raises(ValueError):
        question_validator.validate_question(
            conversation="1234", query="What is the meaning of life?"
        )


def test_valid_responses(question_validator, monkeypatch):
    """Test how valid responses are handled by QuestionValidator."""
    for retval in ["INVALID,NOYAML", "VALID,NOYAML", "VALID,YAML"]:
        ml = mock_llm_chain({"text": retval})
        monkeypatch.setattr(ols.src.query_helpers.question_validator, "LLMChain", ml)
        monkeypatch.setattr(
            ols.src.query_helpers.question_validator, "LLMLoader", mock_llm_loader()
        )

        response = question_validator.validate_question(
            conversation="1234", query="What is the meaning of life?"
        )

        assert response == retval.split(",")
