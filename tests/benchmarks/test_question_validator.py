"""Benchmarks for QuestionValidator class."""

from unittest.mock import patch

from langchain_core.messages import AIMessage

from ols import config
from ols.constants import GenericLLMParameters

# needs to be setup there before is_user_authorized is imported
config.ols_config.authentication_config.module = "k8s"

from ols.src.query_helpers.question_validator import QuestionValidator  # noqa: E402
from tests.mock_classes.mock_llm_loader import mock_llm_loader  # noqa: E402


def perform_question_validation_benchmark(benchmark, question):
    """Prepare all mocks and the call the QuestionValidator.validate_question method."""
    # it is needed to initialize configuration in order to be able
    # to construct QuestionValidator instance
    config.reload_from_yaml_file("tests/config/valid_config.yaml")

    with patch(
        "ols.src.query_helpers.question_validator.QuestionValidator._invoke_llm"
    ) as mock_invoke:
        mock_invoke.return_value = AIMessage(content="ACCEPTED")

        # when the LLM will be initialized the check for provided parameters will
        # be performed
        llm_loader = mock_llm_loader(
            None,
            expected_params=(
                "p1",
                "m1",
                {GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: 4},
            ),
        )

        # check that LLM loader was called with expected parameters
        question_validator = QuestionValidator(llm_loader=llm_loader)

        # just run the validation for the provided question, we just need to check
        # parameters passed to LLM that is performed in mock object
        benchmark(
            question_validator.validate_question,
            "123e4567-e89b-12d3-a456-426614174000",
            question,
        )


def test_validate_question_short_question(benchmark):
    """Benchmarks the method QuestionValidator.validate_question for short question."""
    question = "question"
    perform_question_validation_benchmark(benchmark, question)


def test_validate_question_medium_question(benchmark):
    """Benchmarks the method QuestionValidator.validate_question for medium-sized question."""
    question = (
        "Suppose I start with a newly deployed OpenShift 4.15 cluster, what do I need "
        "to do to send logs from the cluster to AWS CloudWatch? "
        "Is it even possible to do so?"
    )
    perform_question_validation_benchmark(benchmark, question)


def test_validate_question_long_question(benchmark):
    """Benchmarks the method QuestionValidator.validate_question for long question."""
    question = (
        "Suppose I start with a newly deployed OpenShift 4.15 cluster, what do I need "
        "to do to send logs from the cluster to AWS CloudWatch? "
        "Is it even possible to do so?"
    )
    # make sure we don't exceed context window size there
    # it is set to just 450 tokens in the configuration file
    question *= 8
    perform_question_validation_benchmark(benchmark, question)
