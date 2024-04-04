"""Unit tests for QuestionValidator class."""

import pytest

from ols.src.query_helpers.question_validator import QueryHelper, QuestionValidator
from ols.utils import config
from tests.mock_classes.llm_loader import mock_llm_loader


@pytest.fixture
def question_validator():
    """Fixture containing constructed and initialized QuestionValidator."""
    config.init_empty_config()
    return QuestionValidator(llm_loader=mock_llm_loader(None))


def test_is_query_helper_subclass():
    """Test that QuestionValidator is a subclass of QueryHelper."""
    assert issubclass(QuestionValidator, QueryHelper)


def test_passing_parameters():
    """Test that llm_params is handled correctly and without runtime error."""
    question_validator = QuestionValidator()
    assert question_validator.llm_params is not None
    assert "max_new_tokens" in question_validator.llm_params is not None
    assert "min_new_tokens" in question_validator.llm_params is not None

    question_validator = QuestionValidator(llm_params={})
    # the llm_params should be rewritten in constructor
    assert question_validator.llm_params is not None
    assert "max_new_tokens" in question_validator.llm_params is not None
    assert "min_new_tokens" in question_validator.llm_params is not None
