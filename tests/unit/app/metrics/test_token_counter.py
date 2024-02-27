"""Unit tests for GenericTokenCounter class."""

from ols.app.metrics import GenericTokenCounter


class MockLLM:
    """Mocked LLM to be used in unit tests."""

    def get_num_tokens(self, prompt):
        """Poor man's token counter."""
        return len(prompt.split(" "))


def test_on_llm_start():
    """Test the GenericTokenCounter.on_llm_start method."""
    llm = MockLLM()

    # initialize new token counter
    token_counter = GenericTokenCounter(llm)

    # a beginning the counters should be zeroed
    assert token_counter.llm_calls == 0
    assert token_counter.input_tokens_counted == 0

    # token count for empty input
    token_counter.on_llm_start({}, [])

    # token counter needs to be zero as mocked LLM does not process anything
    assert token_counter.llm_calls == 1
    assert token_counter.input_tokens_counted == 0

    # now the prompt will be tokenized into 5 tokens
    token_counter.on_llm_start({}, ["this is just a test"])
    assert token_counter.llm_calls == 2
    assert token_counter.input_tokens_counted == 5
