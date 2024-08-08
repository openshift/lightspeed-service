"""Benchmarks for QuestionValidator class."""

from unittest.mock import patch

from ols import config
from ols.constants import GenericLLMParameters
from ols.src.query_helpers.question_validator import QuestionValidator
from tests.mock_classes.mock_llm_chain import mock_llm_chain
from tests.mock_classes.mock_llm_loader import mock_llm_loader


@patch("ols.src.query_helpers.question_validator.LLMChain", new=mock_llm_chain(None))
def test_validate_question_llm_loader(benchmark):
    """Benchmarks the method QuestionValidator.validate_question."""
    # it is needed to initialize configuration in order to be able
    # to construct QuestionValidator instance
    config.reload_from_yaml_file("tests/config/valid_config.yaml")

    # when the LLM will be initialized the check for provided parameters will
    # be performed
    llm_loader = mock_llm_loader(
        None,
        expected_params=("p1", "m1", {GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: 4}),
    )

    # check that LLM loader was called with expected parameters
    question_validator = QuestionValidator(llm_loader=llm_loader)

    # just run the validation, we just need to check parameters passed to LLM
    # that is performed in mock object
    benchmark(
        question_validator.validate_question,
        "123e4567-e89b-12d3-a456-426614174000",
        "query",
    )
