"""Unit tests for QuestionValidator class."""

from unittest.mock import patch

import pytest

from ols.src.query_helpers.question_validator import QueryHelper, QuestionValidator
from ols.utils import config
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader


@pytest.fixture
def question_validator():
    """Fixture containing constructed and initialized QuestionValidator."""
    config.init_empty_config()
    return QuestionValidator()

def test_is_query_helper_subclass():
    """Test that QuestionValidator is a subclass of QueryHelper."""
    assert issubclass(QuestionValidator, QueryHelper)

@patch("ols.src.query_helpers.question_validator.LLMLoader", new=mock_llm_loader(None))
def test_valid_responses(question_validator):
    """Test how valid responses are handled by QuestionValidator."""
    for retval in ["INVALID,NOYAML", "VALID,NOYAML", "VALID,YAML", "VALID,REPHRASE"]:
        # basically `@patch` and `with patch():` do the same thing, but the latter
        # allow us to change the class/method/function behaviour in runtime
        ml = mock_llm_chain({"text": retval})
        with patch("ols.src.query_helpers.question_validator.LLMChain", new=ml):
            response = question_validator.validate_question(
                conversation="1234", query="What is the meaning of life?"
            )

            assert response == retval.split(",")
