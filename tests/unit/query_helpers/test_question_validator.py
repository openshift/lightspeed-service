"""Unit tests for QuestionValidator class."""

from unittest.mock import patch

import pytest

from ols import config
from ols.constants import GenericLLMParameters

# needs to be setup there before is_user_authorized is imported
config.ols_config.authentication_config.module = "k8s"

from ols.src.query_helpers.question_validator import (  # noqa: E402
    QueryHelper,
    QuestionValidator,
)
from tests.mock_classes.mock_llm_chain import mock_llm_chain  # noqa: E402
from tests.mock_classes.mock_llm_loader import mock_llm_loader  # noqa: E402


@pytest.fixture
def question_validator():
    """Fixture containing constructed and initialized QuestionValidator."""
    return QuestionValidator(llm_loader=mock_llm_loader(None))


def test_is_query_helper_subclass():
    """Test that QuestionValidator is a subclass of QueryHelper."""
    assert issubclass(QuestionValidator, QueryHelper)


def test_passing_parameters():
    """Test that generic_llm_params is handled correctly and without runtime error."""
    # it is needed to initialize configuration in order to be able
    # to construct QuestionValidator instance
    config.reload_from_yaml_file("tests/config/valid_config.yaml")

    question_validator = QuestionValidator()
    assert question_validator.generic_llm_params is not None
    assert (
        GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE
        in question_validator.generic_llm_params
    )
    assert (
        question_validator.generic_llm_params[
            GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE
        ]
        == 4
    )

    question_validator = QuestionValidator(generic_llm_params={})
    # the generic_llm_params should be rewritten in constructor
    assert question_validator.generic_llm_params is not None
    assert (
        GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE
        in question_validator.generic_llm_params
    )
    assert (
        question_validator.generic_llm_params[
            GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE
        ]
        == 4
    )


@patch("ols.src.query_helpers.question_validator.LLMChain", new=mock_llm_chain(None))
def test_validate_question_llm_loader():
    """Test that LLM is loaded within validate_question method with proper parameters."""
    # it is needed to initialize configuration in order to be able
    # to construct QuestionValidator instance
    config.reload_from_yaml_file("tests/config/valid_config.yaml")

    # when the LLM will be initialized the check for provided parameters will
    # be performed
    llm_loader = mock_llm_loader(
        None,
        expected_params=(
            "p1",
            "m1",
            {GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: 4},
            False,
        ),
    )

    # check that LLM loader was called with expected parameters
    question_validator = QuestionValidator(llm_loader=llm_loader)

    # just run the validation, we just need to check parameters passed to LLM
    # that is performed in mock object
    question_validator.validate_question(
        "123e4567-e89b-12d3-a456-426614174000", "query"
    )
