"""Unit tests for GenericTokenCounter class."""

from langchain_core.outputs.llm_result import LLMResult

from ols import config

# needs to be setup there before is_user_authorized is imported
config.ols_config.authentication_config.module = "k8s"


from ols.app.metrics import GenericTokenCounter  # noqa:E402


class MockLLM:
    """Mocked LLM to be used in unit tests."""

    def get_num_tokens(self, prompt):
        """Poor man's token counter."""
        return len(prompt.split(" "))


def test_on_llm_start():
    """Test the GenericTokenCounter.on_llm_start method."""
    llm = MockLLM()

    # initialize new token counter
    generic_token_counter = GenericTokenCounter(llm)

    # a beginning the counters should be zeroed
    assert generic_token_counter.token_counter.llm_calls == 0
    assert generic_token_counter.token_counter.input_tokens_counted == 0

    # check the textual representation as well
    expected = (
        "GenericTokenCounter: input_tokens: 0 output_tokens: 0 counted: 0 LLM calls: 0"
    )
    assert str(generic_token_counter) == expected

    # token count for empty input
    generic_token_counter.on_llm_start({}, [])

    # token counter needs to be zero as mocked LLM does not process anything
    assert generic_token_counter.token_counter.llm_calls == 1
    assert generic_token_counter.token_counter.input_tokens_counted == 0

    # check the textual representation as well
    expected = (
        "GenericTokenCounter: input_tokens: 0 output_tokens: 0 counted: 0 LLM calls: 1"
    )
    assert str(generic_token_counter) == expected

    # now the prompt will be tokenized into 5 tokens
    generic_token_counter.on_llm_start({}, ["this is just a test"])
    assert generic_token_counter.token_counter.llm_calls == 2
    assert generic_token_counter.token_counter.input_tokens_counted == 5

    # check the textual representation as well
    expected = (
        "GenericTokenCounter: input_tokens: 0 output_tokens: 0 counted: 5 LLM calls: 2"
    )
    assert str(generic_token_counter) == expected


class MockResult:
    """Mocked one result that can be made part of LLMResult."""

    def __init__(self, prompt_tokens, completion_tokens):
        """Perform setup of one result, initializing it's tokens counters."""
        self.llm_output = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        }


class MockLLMResult:
    """Mocked LLMResult value."""

    def __init__(self, results):
        """Perform setup of LLMResult mocked object."""
        self.results = results

    def flatten(self) -> list[LLMResult]:
        """Return results that are already flattened at input."""
        return self.results


def test_on_llm_end():
    """Test the GenericTokenCounter.on_llm_end method."""
    llm = MockLLM()

    # initialize new token counter
    generic_token_counter = GenericTokenCounter(llm)
    assert generic_token_counter.token_counter.input_tokens == 0
    assert generic_token_counter.token_counter.output_tokens == 0

    # check the textual representation as well
    expected = (
        "GenericTokenCounter: input_tokens: 0 output_tokens: 0 counted: 0 LLM calls: 0"
    )
    assert str(generic_token_counter) == expected

    # empty response
    response = MockLLMResult([])
    generic_token_counter.on_llm_end(response)

    # for empty response, counters should not change
    assert generic_token_counter.token_counter.input_tokens == 0
    assert generic_token_counter.token_counter.output_tokens == 0

    # check the textual representation as well
    expected = (
        "GenericTokenCounter: input_tokens: 0 output_tokens: 0 counted: 0 LLM calls: 0"
    )
    assert str(generic_token_counter) == expected

    # non-empty response
    x = MockResult(10, 20)
    response = MockLLMResult([x])
    generic_token_counter.on_llm_end(response)

    # for non-empty response, counters should change
    assert generic_token_counter.token_counter.input_tokens == 10
    assert generic_token_counter.token_counter.output_tokens == 20

    # check the textual representation as well
    expected = "GenericTokenCounter: input_tokens: 10 output_tokens: 20 counted: 0 LLM calls: 0"
    assert str(generic_token_counter) == expected
