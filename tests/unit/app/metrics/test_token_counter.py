"""Unit tests for GenericTokenCounter class."""

import pytest

from ols import config

# needs to be setup there before is_user_authorized is imported
config.ols_config.authentication_config.module = "k8s"


from ols.app.metrics import GenericTokenCounter  # noqa:E402


class MockLLM:
    """Mocked LLM to be used in unit tests."""

    def get_num_tokens(self, prompt):
        """Poor man's token counter."""
        return len(prompt.split(" "))


@pytest.mark.asyncio
async def test_on_llm_start():
    """Test the GenericTokenCounter.on_llm_start method."""
    llm = MockLLM()

    # initialize new token counter
    generic_token_counter = GenericTokenCounter(llm)

    # a beginning the counters should be zeroed
    assert generic_token_counter.token_counter.llm_calls == 0
    assert generic_token_counter.token_counter.input_tokens == 0

    # check the textual representation as well
    expected = "GenericTokenCounter: input_tokens: 0 output_tokens: 0 LLM calls: 0"
    assert str(generic_token_counter) == expected

    # token count for empty input
    await generic_token_counter.on_llm_start({}, [])

    # token counter needs to be zero as mocked LLM does not process anything
    assert generic_token_counter.token_counter.llm_calls == 1
    assert generic_token_counter.token_counter.input_tokens == 0

    # check the textual representation as well
    expected = "GenericTokenCounter: input_tokens: 0 output_tokens: 0 LLM calls: 1"
    assert str(generic_token_counter) == expected

    # now the prompt will be tokenized into 5 tokens
    await generic_token_counter.on_llm_start({}, ["this is just a test"])
    assert generic_token_counter.token_counter.llm_calls == 2
    assert generic_token_counter.token_counter.input_tokens == 5

    # check the textual representation as well
    expected = "GenericTokenCounter: input_tokens: 5 output_tokens: 0 LLM calls: 2"
    assert str(generic_token_counter) == expected


@pytest.mark.asyncio
async def test_on_llm_end():
    """Test the GenericTokenCounter.on_llm_new_token method."""
    llm = MockLLM()

    # initialize new token counter
    generic_token_counter = GenericTokenCounter(llm)
    assert generic_token_counter.token_counter.input_tokens == 0
    assert generic_token_counter.token_counter.output_tokens == 0

    # check the textual representation as well
    expected = "GenericTokenCounter: input_tokens: 0 output_tokens: 0 LLM calls: 0"
    assert str(generic_token_counter) == expected

    # empty token
    await generic_token_counter.on_llm_new_token("")
    await generic_token_counter.on_llm_new_token(None)

    # for empty response, counters should not change
    assert generic_token_counter.token_counter.input_tokens == 0
    assert generic_token_counter.token_counter.output_tokens == 0

    # check the textual representation as well
    expected = "GenericTokenCounter: input_tokens: 0 output_tokens: 0 LLM calls: 0"
    assert str(generic_token_counter) == expected

    # non-empty response
    await generic_token_counter.on_llm_new_token("hello")
    await generic_token_counter.on_llm_new_token("there")

    # for non-empty response, counters should change
    assert generic_token_counter.token_counter.input_tokens == 0
    assert generic_token_counter.token_counter.output_tokens == 2

    # check the textual representation as well
    expected = "GenericTokenCounter: input_tokens: 0 output_tokens: 2 LLM calls: 0"
    assert str(generic_token_counter) == expected
