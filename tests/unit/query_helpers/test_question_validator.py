import pytest

import src.query_helpers.question_validator
from src.query_helpers.question_validator import QuestionValidator
from tests.mock_classes.llm_chain import mock_llm_chain


@pytest.fixture
def question_validator():
    return QuestionValidator()


def test_invalid_response(question_validator, monkeypatch):
    # response not in the following set should generate a ValueError
    # [INVALID,NOYAML]
    # [VALID,NOYAML]
    # [VALID,YAML]

    ml = mock_llm_chain({"text": "default"})
    monkeypatch.setattr(src.query_helpers.question_validator, "LLMChain", ml)

    with pytest.raises(ValueError):
        question_validator.validate_question(
            conversation="1234", query="What is the meaning of life?"
        )


def test_valid_responses(question_validator, monkeypatch):
    for retval in ["INVALID,NOYAML", "VALID,NOYAML", "VALID,YAML"]:
        ml = mock_llm_chain({"text": retval})
        monkeypatch.setattr(src.query_helpers.question_validator, "LLMChain", ml)

        response = question_validator.validate_question(
            conversation="1234", query="What is the meaning of life?"
        )

        assert response == retval.split(",")
